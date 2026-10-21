import uuid
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Annotated
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db, set_transaction_rls_user
from app.models.account import Account
from app.models.session import AuthSession

settings = get_settings()

ph = PasswordHasher(
    memory_cost=settings.ARGON2_MEMORY_COST_KIB,
    time_cost=settings.ARGON2_TIME_COST,
    parallelism=settings.ARGON2_PARALLELISM
)

security_scheme = HTTPBearer(auto_error=False)

def validate_password_strength(password: str) -> None:
    """Validate the shared 6-128 character account password policy."""
    if len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords must contain at least 6 characters."
        )
    if len(password) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords must contain no more than 128 characters."
        )

def normalize_email(email: str) -> str:
    """Normalize an email for case-insensitive account lookup and persistence."""
    return email.strip().lower()

def hash_password(password: str) -> str:
    validate_password_strength(password)
    return ph.hash(password)

_dummy_password_hash: str | None = None


def dummy_password_hash() -> str:
    """A real Argon2 hash of a random secret, for emails that have no account.

    Checking the password against it costs as much as checking a real one, so
    how long sign-in takes does not reveal whether the email is registered.
    """
    global _dummy_password_hash
    if _dummy_password_hash is None:
        _dummy_password_hash = ph.hash(secrets.token_urlsafe(32))
    return _dummy_password_hash


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def generate_refresh_token() -> str:
    return secrets.token_hex(32)

def create_access_token(account_id: str, session_id: str, role: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    jti = str(uuid.uuid4())
    payload = {
        "sub": str(account_id),
        "sid": str(session_id),
        "jti": jti,
        "role": role,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(exp.timestamp())
    }
    encoded_jwt = jwt.encode(payload, settings.JWT_PRIVATE_KEY, algorithm="RS256")
    return encoded_jwt, jti

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY,
            algorithms=["RS256"],
            issuer=settings.JWT_ISSUER,
            audience=settings.JWT_AUDIENCE
        )
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}"
        )

async def get_current_user_and_session(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> tuple[Account, AuthSession]:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header."
        )

    payload = decode_access_token(credentials.credentials)
    account_id_str = payload.get("sub")
    session_id_str = payload.get("sid")

    if not account_id_str or not session_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token claims."
        )

    try:
        account_id = uuid.UUID(account_id_str)
        session_id = uuid.UUID(session_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token identifiers.")

    account_stmt = select(Account).where(Account.id == account_id)
    account_res = await db.execute(account_stmt)
    account = account_res.scalar_one_or_none()

    if not account or not account.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account disabled or not found.")

    session_stmt = select(AuthSession).where(AuthSession.id == session_id, AuthSession.owner_id == account_id)
    session_res = await db.execute(session_stmt)
    session = session_res.scalar_one_or_none()

    if not session or session.is_revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or revoked.")

    now = datetime.now(timezone.utc)
    session_expires_at = session.expires_at
    if session_expires_at.tzinfo is None:
        session_expires_at = session_expires_at.replace(tzinfo=timezone.utc)

    if session_expires_at < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or revoked.")

    await set_transaction_rls_user(db, account.id)

    return account, session
