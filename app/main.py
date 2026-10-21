from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine, Base
import app.models  # noqa: F401
from app.api.v1 import auth, sync, ai, meetings, meeting_notes
from app.admin import setup_admin
from app.clients.deepseek import DeepSeekClient
from app.clients.gemini_tts import GeminiTtsClient
from app.clients.pinecone_index import PineconeIndexClient
from app.services.handbook_rag import HandbookRetriever

settings = get_settings()
deepseek_client = DeepSeekClient(settings=settings)
gemini_tts_client = GeminiTtsClient(settings=settings)
handbook_retriever = HandbookRetriever(settings, PineconeIndexClient(settings))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize HTTP client pools
    await deepseek_client.start()
    await gemini_tts_client.start()
    await handbook_retriever.start()
    app.state.deepseek_client = deepseek_client
    app.state.gemini_tts_client = gemini_tts_client
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
    await gemini_tts_client.close()
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
app.state.gemini_tts_client = gemini_tts_client
app.state.handbook_retriever = handbook_retriever

# Paths that carry an uploaded document rather than a record, and so are held
# to the upload ceiling instead of the 1 MiB one.
UPLOAD_PATHS = ("/v1/ai/flashcards", "/v1/ai/study-notes")


# Enforce max body size middleware
@app.middleware("http")
async def limit_body_size_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    is_upload = request.url.path in UPLOAD_PATHS
    limit = settings.MAX_UPLOAD_BODY_SIZE_BYTES if is_upload else settings.MAX_BODY_SIZE_BYTES
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid Content-Length header."},
            )
        if declared > limit:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": (
                        f"Request payload exceeds maximum allowed size of "
                        f"{limit // (1024 * 1024)} MiB."
                    )
                },
            )
    return await call_next(request)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Disable CORS by default unless explicitly enabled
if settings.ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include Personal Routers (business & Gmail routes removed in this version)
app.include_router(auth.router)
app.include_router(sync.router)
app.include_router(ai.router)
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
