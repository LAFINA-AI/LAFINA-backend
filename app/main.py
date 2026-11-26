from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine, Base
import app.models  # noqa: F401
from app.api.v1 import auth, sync, ai, meetings, meeting_notes, voice
from app.admin import setup_admin
from app.clients.deepseek import DeepSeekClient
from app.clients.elevenlabs_tts import ElevenLabsTtsClient
from app.clients.elevenlabs_agents import ElevenLabsAgentClient
from app.clients.pinecone_index import PineconeIndexClient
from app.services.handbook_rag import HandbookRetriever

settings = get_settings()
deepseek_client = DeepSeekClient(settings=settings)
elevenlabs_tts_client = ElevenLabsTtsClient(settings=settings)
elevenlabs_agent_client = ElevenLabsAgentClient(settings=settings)
handbook_retriever = HandbookRetriever(settings, PineconeIndexClient(settings))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize HTTP client pools
    await deepseek_client.start()
    await elevenlabs_tts_client.start()
    await elevenlabs_agent_client.start()
    await handbook_retriever.start()
    app.state.deepseek_client = deepseek_client
    app.state.elevenlabs_tts_client = elevenlabs_tts_client
    app.state.elevenlabs_agent_client = elevenlabs_agent_client
    app.state.handbook_retriever = handbook_retriever
    if not settings.is_handbook_configured():
        print(
            f"[Info] Student Handbook RAG is off: {settings.get_pinecone_invalid_reason()}"
        )

    # On Vercel serverless, schema migrations run at deploy time over the direct
    # (unpooled) Neon connection via `alembic upgrade head`, keeping cold starts fast.
    if not settings.is_vercel_runtime():
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Feature flag rows appear in the admin panel ready to flip.
            from app.database import AsyncSessionLocal
            from app.services.feature_flags import ensure_default_flags
            async with AsyncSessionLocal() as flag_session:
                await ensure_default_flags(flag_session)

            if settings.ADMIN_EMAIL and settings.ADMIN_PASSWORD:
                from scripts.create_admin import create_admin
                await create_admin(settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD)
        except Exception as e:
            print(f"[Warning] Startup initialization note: {e}")
    yield
    await deepseek_client.close()
    await elevenlabs_tts_client.close()
    await elevenlabs_agent_client.close()
    await handbook_retriever.close()
    await engine.dispose()


app = FastAPI(
    title="LAFINA Cloud API",
    description="FastAPI Backend for LAFINA Offline-First Voice Scheduler (Vercel + Neon Postgres)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url=None,
)

app.state.deepseek_client = deepseek_client
app.state.elevenlabs_tts_client = elevenlabs_tts_client
app.state.elevenlabs_agent_client = elevenlabs_agent_client
app.state.handbook_retriever = handbook_retriever

# Paths that carry an uploaded document rather than a record, and so are held
# to the upload ceiling instead of the 1 MiB one.
UPLOAD_PATHS = ("/v1/ai/flashcards", "/v1/ai/study-notes")


class _BodyTooLargeError(Exception):
    pass


class MaxBodySizeMiddleware:
    """Pure ASGI body-size limiter enforcing both Content-Length and chunked streams."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        is_upload = path in UPLOAD_PATHS
        limit = settings.MAX_UPLOAD_BODY_SIZE_BYTES if is_upload else settings.MAX_BODY_SIZE_BYTES
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length.decode("latin-1").strip())
            except (ValueError, UnicodeDecodeError):
                response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return
            if declared < 0:
                response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return
            if declared > limit:
                response = JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "detail": (
                            f"Request payload exceeds maximum allowed size of "
                            f"{limit // (1024 * 1024)} MiB."
                        )
                    },
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        response_started = False
        too_large = False

        async def limited_receive():
            nonlocal received_bytes, too_large
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                received_bytes += len(chunk)
                if received_bytes > limit:
                    too_large = True
                    raise _BodyTooLargeError()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if too_large:
                # Suppress FastAPI's generic 400 body-parse error so we emit 413 cleanly.
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            pass

        if too_large and not response_started:
            response = JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={
                    "detail": (
                        f"Request payload exceeds maximum allowed size of "
                        f"{limit // (1024 * 1024)} MiB."
                    )
                },
            )
            await response(scope, receive, send)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Register streaming body-size limiter outermost so chunked oversized requests
# are stopped before Starlette/FastAPI buffers them.
app.add_middleware(MaxBodySizeMiddleware)


# Disable CORS by default unless explicitly enabled
if settings.ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include Personal Routers (business & Gmail routes removed in this version)
app.include_router(auth.router)
app.include_router(sync.router)
app.include_router(ai.router)
app.include_router(voice.router)
app.include_router(meeting_notes.router)
app.include_router(meetings.router)

# Mount SQLAdmin UI
setup_admin(app, engine)


@app.get("/v1/me", response_model=auth.UserProfileResponse, tags=["auth"])
async def get_me_top_level(
    auth_data: tuple[auth.Account, auth.AuthSession] = Depends(auth.get_current_user_and_session),
):
    account, _ = auth_data
    return auth.build_user_profile(account)


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    return {"status": "ok", "environment": settings.ENVIRONMENT, "version": "1.0.0"}
