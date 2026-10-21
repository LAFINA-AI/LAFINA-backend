"""Personal-plan entitlements.

This version of LAFINA has no business role, workspaces or seats, so every
entitlement is decided by the account row alone: its `role`, `system_role`
and `subscription_plan`.
"""
from pydantic import BaseModel

from app.models.account import Account

# Plans that unlock the online (paid) features.
PRO_PLAN = "student_pro"


class CapabilityResolution(BaseModel):
    system_role: str
    subscription_plan: str
    # Kept for API compatibility with older clients; without business seats
    # the effective plan is always the account's own plan.
    effective_subscription_plan: str


def is_admin(account: Account) -> bool:
    """True for platform administrators (either admin flag)."""
    return account.role == "admin" or (account.system_role or "user") == "admin"


def is_student_pro(account: Account) -> bool:
    """The Student Pro plan itself. Admin accounts are not included."""
    return bool(account.is_active) and (
        account.role == PRO_PLAN or account.subscription_plan == PRO_PLAN
    )


def has_online_ai_access(account: Account) -> bool:
    """Online AI (chat, flashcards, study notes, meeting notes): Student Pro or admin."""
    return bool(account.is_active) and (is_student_pro(account) or is_admin(account))


def has_tts_access(account: Account) -> bool:
    """Cloud TTS is a Student Pro feature only (admins are deliberately excluded)."""
    return is_student_pro(account)


def resolve_account_capabilities(account: Account) -> CapabilityResolution:
    """Describe the account's plan for the `/me` profile response."""
    base_plan = account.subscription_plan or "student"
    return CapabilityResolution(
        system_role=account.system_role or "user",
        subscription_plan=base_plan,
        effective_subscription_plan=base_plan,
    )
