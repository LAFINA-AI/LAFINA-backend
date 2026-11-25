import secrets
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, update

from app.config import get_settings
from app.database import get_db
from app.models.account import Account
from app.models.session import AuthSession
from app.models.recovery import RecoveryCode
from app.security.auth import (
    hash_password, verify_password, needs_password_rehash, create_access_token,
    generate_refresh_token, hash_token, get_current_user_and_session,
    normalize_email, validate_password_strength, dummy_password_hash,
    client_ip_from_request,
)
from app.security import login_throttle

from app.services.capabilities import resolve_account_capabilities

router = APIRouter(prefix="/v1/auth", tags=["auth"])
settings = get_settings()

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_info: str | None = Field(default=None, max_length=255)

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1, max_length=256)

class RecoverRequest(BaseModel):
    email: EmailStr
    recovery_code: str = Field(..., min_length=1, max_length=64)
    new_password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900
    user_id: str
    email: str
    role: str
    system_role: str = "user"
    subscription_plan: str = "student"
    recovery_codes: list[str] | None = None

class UserProfileResponse(BaseModel):
    id: str
    email: str
    role: str
    system_role: str = "user"
    subscription_plan: str = "student"
    effective_subscription_plan: str = "student"
    is_active: bool
    created_at: str

@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    validate_password_strength(req.password)
    now = datetime.now(timezone.utc)
    client_ip = client_ip_from_request(request)
    if client_ip:
        reg_key = login_throttle.ip_key(client_ip)
        retry_after = await login_throttle.seconds_until_allowed(
            db,
            reg_key,
            now,
            event_type=login_throttle.REGISTER_IP_EVENT,
            limit=settings.MAX_REGISTRATIONS_PER_IP_PER_HOUR,
            window=login_throttle.REGISTER_WINDOW,
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registrations from this network. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )

    normalized_email = normalize_email(str(req.email))
    stmt = select(Account).where(func.lower(Account.email) == normalized_email)
    res = await db.execute(stmt)
    if res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account with this email already exists.")

    password_hash = hash_password(req.password)
    account = Account(email=normalized_email, password_hash=password_hash, role="student")
    db.add(account)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account with this email already exists.",
        )

    raw_refresh = generate_refresh_token()
    refresh_hash = hash_token(raw_refresh)
    expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    session = AuthSession(
        owner_id=account.id,
        refresh_token_hash=refresh_hash,
        expires_at=expires_at
    )
    db.add(session)
    await db.flush()

    # Generate 4 one-time 80-bit recovery codes
    raw_recovery_codes = [secrets.token_hex(10).upper() for _ in range(4)]
    for rec_code in raw_recovery_codes:
        db.add(RecoveryCode(owner_id=account.id, code_hash=hash_token(rec_code)))

    if client_ip:
        from app.models.ai_usage import SecurityEvent
        db.add(
            SecurityEvent(
                event_type=login_throttle.REGISTER_IP_EVENT,
                owner_id=account.id,
                ip_address=client_ip[:64],
                details=login_throttle.ip_key(client_ip),
            )
        )

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account with this email already exists.",
        )

    access_token, _ = create_access_token(str(account.id), str(session.id), account.role)

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(account.id),
        email=account.email,
        role=account.role,
        system_role=account.system_role,
        subscription_plan=account.subscription_plan,
        recovery_codes=raw_recovery_codes
    )

@router.post("/login", response_model=AuthTokenResponse)
async def login(req: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    validate_password_strength(req.password, check_blocklist=False)
    normalized_email = normalize_email(str(req.email))
    client_ip = client_ip_from_request(request)

    # Refused before the password is looked at, so a guesser learns nothing
    # more once the limit is reached, and the answer is the same whether or
    # not the email has an account.
    throttle_key = login_throttle.email_key(normalized_email)
    retry_after = await login_throttle.seconds_until_allowed(
        db, throttle_key, datetime.now(timezone.utc), ip_address=client_ip
    )
    if retry_after is not None:
        minutes = max(1, -(-retry_after // 60))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Too many sign-in attempts for this email. "
                f"Try again in {minutes} minute{'s' if minutes != 1 else ''}."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    stmt = select(Account).where(func.lower(Account.email) == normalized_email)
    res = await db.execute(stmt)
    account = res.scalar_one_or_none()

    # An unknown email is checked against a stand-in hash, so it takes as long
    # to refuse as a wrong password and the timing does not reveal it.
    password_ok = verify_password(
        req.password, account.password_hash if account else dummy_password_hash()
    )
    if not account or not password_ok:
        await login_throttle.record_failure(
            db,
            throttle_key,
            owner_id=account.id if account else None,
            ip_address=client_ip,
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    if not account.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")

    if needs_password_rehash(account.password_hash):
        account.password_hash = hash_password(req.password)

    raw_refresh = generate_refresh_token()
    refresh_hash = hash_token(raw_refresh)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    session = AuthSession(
        owner_id=account.id,
        refresh_token_hash=refresh_hash,
        device_info=req.device_info,
        expires_at=expires_at
    )
    db.add(session)
    await db.commit()

    access_token, _ = create_access_token(str(account.id), str(session.id), account.role)

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(account.id),
        email=account.email,
        role=account.role,
        system_role=account.system_role,
        subscription_plan=account.subscription_plan,
    )

@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    old_hash = hash_token(req.refresh_token)
    stmt = select(AuthSession).where(AuthSession.refresh_token_hash == old_hash)
    res = await db.execute(stmt)
    old_session = res.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if not old_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    if old_session.is_revoked:
        # Record reuse of a revoked refresh token for security auditing without
        # breaking already-rotated sessions on flaky mobile retries.
        from app.models.ai_usage import SecurityEvent
        db.add(
            SecurityEvent(
                event_type="refresh_token_reuse",
                owner_id=old_session.owner_id,
                details=f"session:{old_session.id}",
            )
        )
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    old_expires_at = old_session.expires_at
    if old_expires_at.tzinfo is None:
        old_expires_at = old_expires_at.replace(tzinfo=timezone.utc)

    if old_expires_at < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    # Atomically claim & revoke old session so concurrent refresh races cannot duplicate sessions.
    claimed = await db.execute(
        update(AuthSession)
        .where(AuthSession.id == old_session.id, AuthSession.is_revoked.is_(False))
        .values(is_revoked=True)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    account_stmt = select(Account).where(Account.id == old_session.owner_id)
    account_res = await db.execute(account_stmt)
    account = account_res.scalar_one_or_none()

    if not account or not account.is_active:
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account inactive.")

    # Issue new refresh token & session
    new_raw_refresh = generate_refresh_token()
    new_refresh_hash = hash_token(new_raw_refresh)
    new_expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    new_session = AuthSession(
        owner_id=account.id,
        refresh_token_hash=new_refresh_hash,
        device_info=old_session.device_info,
        expires_at=new_expires_at
    )
    db.add(new_session)
    await db.commit()

    access_token, _ = create_access_token(str(account.id), str(new_session.id), account.role)

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=new_raw_refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(account.id),
        email=account.email,
        role=account.role,
        system_role=account.system_role,
        subscription_plan=account.subscription_plan,
    )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    auth_data: tuple[Account, AuthSession] = Depends(get_current_user_and_session),
    db: AsyncSession = Depends(get_db)
):
    _, session = auth_data
    session.is_revoked = True
    await db.commit()
    return None

@router.post("/recover", status_code=status.HTTP_200_OK)
async def recover(req: RecoverRequest, request: Request, db: AsyncSession = Depends(get_db)):
    validate_password_strength(req.new_password)

    normalized_email = normalize_email(str(req.email))
    now = datetime.now(timezone.utc)
    client_ip = client_ip_from_request(request)
    throttle_key = login_throttle.email_key(normalized_email)
    retry_after = await login_throttle.seconds_until_allowed(
        db,
        throttle_key,
        now,
        event_type=login_throttle.RECOVER_FAILED_EVENT,
        limit=settings.MAX_LOGIN_FAILURES_PER_15MIN,
    )
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many recovery attempts for this email. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    account_stmt = select(Account).where(func.lower(Account.email) == normalized_email)
    account_res = await db.execute(account_stmt)
    account = account_res.scalar_one_or_none()

    cleaned_code = req.recovery_code.strip().upper()
    rec_hash = hash_token(cleaned_code)
    rec_code = None
    if account and account.is_active:
        rec_stmt = select(RecoveryCode).where(
            RecoveryCode.owner_id == account.id,
            RecoveryCode.code_hash == rec_hash,
            RecoveryCode.is_used.is_(False)
        )
        rec_res = await db.execute(rec_stmt)
        rec_code = rec_res.scalar_one_or_none()

    if not account or not account.is_active or not rec_code:
        await login_throttle.record_failure(
            db,
            throttle_key,
            owner_id=account.id if account else None,
            ip_address=client_ip,
            event_type=login_throttle.RECOVER_FAILED_EVENT,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or recovery code.",
        )

    # Atomically claim the one-time recovery code to prevent concurrent reuse.
    claimed = await db.execute(
        update(RecoveryCode)
        .where(RecoveryCode.id == rec_code.id, RecoveryCode.is_used.is_(False))
        .values(is_used=True, used_at=now)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or recovery code.",
        )

    account.password_hash = hash_password(req.new_password)

    # Revoke all active sessions
    await db.execute(
        update(AuthSession)
        .where(AuthSession.owner_id == account.id)
        .values(is_revoked=True)
    )

    await db.commit()
    return {"detail": "Password successfully reset. Please log in with your new password."}

@router.post("/password", status_code=status.HTTP_200_OK)
async def change_password(
    req: ChangePasswordRequest,
    request: Request,
    auth_data: tuple[Account, AuthSession] = Depends(get_current_user_and_session),
    db: AsyncSession = Depends(get_db),
):
    """Changes the password of the signed-in account.

    The current password is required, so a borrowed access token alone cannot
    lock the owner out. Every other session is revoked because a password
    change is what someone does when they think a device is compromised; the
    session making the change survives so the app stays signed in.
    """
    account, session = auth_data
    now = datetime.now(timezone.utc)
    throttle_key = login_throttle.email_key(account.email)
    retry_after = await login_throttle.seconds_until_allowed(
        db,
        throttle_key,
        now,
        event_type=login_throttle.PASSWORD_CHANGE_FAILED_EVENT,
        limit=settings.MAX_LOGIN_FAILURES_PER_15MIN,
    )
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed password attempts. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    # 400 rather than 401: the request itself is authenticated, and a client
    # that reads 401 as "session expired" would sign the user out over a typo.
    if not verify_password(req.current_password, account.password_hash):
        await login_throttle.record_failure(
            db,
            throttle_key,
            owner_id=account.id,
            ip_address=client_ip_from_request(request),
            event_type=login_throttle.PASSWORD_CHANGE_FAILED_EVENT,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    validate_password_strength(req.new_password)

    if verify_password(req.new_password, account.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The new password must be different from the current one.",
        )

    account.password_hash = hash_password(req.new_password)

    await db.execute(
        update(AuthSession)
        .where(AuthSession.owner_id == account.id, AuthSession.id != session.id)
        .values(is_revoked=True)
    )

    await db.commit()
    return {"detail": "Password changed. Other devices will need the new password."}

def build_user_profile(account: Account) -> UserProfileResponse:
    """Profile payload shared by `/v1/auth/me` and the top-level `/v1/me`."""
    cap_res = resolve_account_capabilities(account)
    return UserProfileResponse(
        id=str(account.id),
        email=account.email,
        role=account.role,
        system_role=cap_res.system_role,
        subscription_plan=cap_res.subscription_plan,
        effective_subscription_plan=cap_res.effective_subscription_plan,
        is_active=account.is_active,
        created_at=account.created_at.isoformat(),
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_me(
    auth_data: tuple[Account, AuthSession] = Depends(get_current_user_and_session),
):
    account, _ = auth_data
    return build_user_profile(account)
