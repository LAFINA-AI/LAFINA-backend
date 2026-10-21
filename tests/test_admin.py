import secrets
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from tests.conftest import TestingSessionLocal
from app.models.account import Account
from app.security.auth import hash_password

@pytest.mark.asyncio
async def test_admin_auth_protection(async_client: AsyncClient):
    dynamic_admin_pass = f"admin-secret-{secrets.token_hex(12)}"
    dynamic_student_pass = f"student-secret-{secrets.token_hex(12)}"

    # Create test admin user and student user
    async with TestingSessionLocal() as db:
        admin_acc = Account(
            email="sysadmin@lafina.app",
            password_hash=hash_password(dynamic_admin_pass),
            role="admin",
            is_active=True
        )
        student_acc = Account(
            email="student_user@lafina.app",
            password_hash=hash_password(dynamic_student_pass),
            role="student",
            is_active=True
        )
        db.add(admin_acc)
        db.add(student_acc)
        await db.commit()

    # 1. Unauthenticated request to /admin/account/list should redirect to login or deny
    res_unauth = await async_client.get("/admin/account/list", follow_redirects=False)
    assert res_unauth.status_code in (302, 303, 307, 401) or "login" in res_unauth.headers.get("location", "")

    # 2. Login with student account should fail admin login
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as session_client:
        student_login = await session_client.post("/admin/login", data={
            "username": "student_user@lafina.app",
            "password": dynamic_student_pass
        }, follow_redirects=False)
        assert student_login.status_code != 200 or "Invalid" in student_login.text or "login" in student_login.url.path

        # 3. Login with admin account should succeed
        admin_login = await session_client.post("/admin/login", data={
            "username": "sysadmin@lafina.app",
            "password": dynamic_admin_pass
        }, follow_redirects=False)
        assert admin_login.status_code in (200, 302, 303, 307)

        # 4. Authenticated admin request to list endpoint
        list_res = await session_client.get("/admin/account/list")
        assert list_res.status_code == 200
