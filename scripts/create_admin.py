import argparse
import asyncio
import getpass
import os
import sys

# Ensure repository root is on sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import func, select
from app.database import AsyncSessionLocal, engine
from app.models.account import Account
from app.security.auth import hash_password, normalize_email, validate_password_strength


async def create_admin(email: str, password: str, *, reset_password: bool = False):
    """Create or promote an admin account without overwriting an existing password on restart."""
    normalized = normalize_email(email)
    if not normalized:
        raise ValueError("Admin email must not be empty.")
    validate_password_strength(password)

    async with AsyncSessionLocal() as db:
        stmt = select(Account).where(func.lower(Account.email) == normalized)
        res = await db.execute(stmt)
        existing = res.scalar_one_or_none()

        if existing:
            existing.role = "admin"
            existing.system_role = "admin"
            existing.is_active = True
            if reset_password:
                existing.password_hash = hash_password(password)
                print(f"[Admin CLI] Updated existing account '{normalized}' to admin with new password.")
            else:
                print(f"[Admin CLI] Verified admin role on existing account '{normalized}' (password kept).")
        else:
            pwd_hash = hash_password(password)
            admin_acc = Account(
                email=normalized,
                password_hash=pwd_hash,
                role="admin",
                system_role="admin",
                is_active=True,
            )
            db.add(admin_acc)
            print(f"[Admin CLI] Created new admin account '{normalized}'.")
        await db.commit()


async def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Create or promote a LAFINA admin account.")
    parser.add_argument("email", nargs="?", help="Admin email address")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="Overwrite password if the admin account already exists.",
    )
    args = parser.parse_args()
    email_input = (args.email or os.environ.get("ADMIN_EMAIL") or input("Enter admin email: ")).strip()
    password_input = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Enter admin password (min 6 chars): ")
    try:
        await create_admin(email_input, password_input, reset_password=args.reset_password)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_run_cli())
