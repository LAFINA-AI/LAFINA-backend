import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.account import Account
from app.security.auth import hash_password

DEMO_PRO_ID = uuid.UUID("11111111-1111-4111-a111-111111111111")
DEMO_ALICE_ID = uuid.UUID("22222222-2222-4222-a222-222222222222")
DEMO_BOB_ID = uuid.UUID("33333333-3333-4333-a333-333333333333")

DEMO_PASSWORD = "Password123!"

DEMO_ACCOUNTS = [
    {
        "id": DEMO_PRO_ID,
        "email": "pro@lafina.ph",
        "role": "student_pro",
        "system_role": "user",
        "subscription_plan": "student_pro",
        "name": "Eleanor Vance (Student Pro)",
    },
    {
        "id": DEMO_ALICE_ID,
        "email": "alice@lafina.ph",
        "role": "student",
        "system_role": "user",
        "subscription_plan": "student",
        "name": "Alice Guo (Student)",
    },
    {
        "id": DEMO_BOB_ID,
        "email": "bob@lafina.ph",
        "role": "student",
        "system_role": "user",
        "subscription_plan": "student",
        "name": "Bob Santos (Student)",
    },
]


async def seed_demo_accounts() -> None:
    """Seed personal student and student_pro demo accounts."""
    print("[Seed] Connecting to database...")
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        pwd_hash = hash_password(DEMO_PASSWORD)

        for acc_info in DEMO_ACCOUNTS:
            stmt = select(Account).where(Account.email == acc_info["email"])
            res = await db.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.password_hash = pwd_hash
                existing.role = acc_info["role"]
                existing.system_role = acc_info["system_role"]
                existing.subscription_plan = acc_info["subscription_plan"]
                existing.is_active = True
                print(f"[Seed] Updated existing account '{acc_info['email']}'.")
            else:
                account = Account(
                    id=acc_info["id"],
                    email=acc_info["email"],
                    password_hash=pwd_hash,
                    role=acc_info["role"],
                    system_role=acc_info["system_role"],
                    subscription_plan=acc_info["subscription_plan"],
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
                db.add(account)
                print(f"[Seed] Created new account '{acc_info['email']}'.")

        await db.commit()
        print("\n========================================================")
        print("DEMO ACCOUNTS SEEDED SUCCESSFULLY")
        print("========================================================")
        print(f"Password for all accounts: {DEMO_PASSWORD}")
        for idx, acc_info in enumerate(DEMO_ACCOUNTS, start=1):
            print(f"\n{idx}. {acc_info['name']}:")
            print(f"   - Email: {acc_info['email']}")
            print(f"   - Plan:  {acc_info['subscription_plan']}")
        print("========================================================\n")


if __name__ == "__main__":
    asyncio.run(seed_demo_accounts())
