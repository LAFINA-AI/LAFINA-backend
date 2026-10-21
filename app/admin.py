import uuid
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse
from sqlalchemy import select

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.models import (
    Account, AuthSession, RecoveryCode,
    ProfileSync, TasksSync, EventsSync, TimeBlocksSync,
    RemindersSync, NotesSync, CustomCategoriesSync,
    PomodoroSettingsSync, PomodoroSessionsSync, FlashcardDecksSync,
    StudySummariesSync, RecordedMeetingsSync,
    IdempotentMutation, ChangeFeed, SyncHead, AIUsage, SecurityEvent,
    FeatureFlag,
)
from app.security.auth import verify_password

settings = get_settings()

_admin_session_maker = None

def set_admin_session_maker(maker):
    global _admin_session_maker
    _admin_session_maker = maker

def get_admin_session_maker():
    global _admin_session_maker
    if _admin_session_maker is not None:
        return _admin_session_maker
    return AsyncSessionLocal

class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        email = str(form.get("username", "")).lower().strip()
        password = str(form.get("password", ""))

        if not email or not password:
            return False

        async with get_admin_session_maker()() as db:
            stmt = select(Account).where(Account.email == email)
            res = await db.execute(stmt)
            account = res.scalar_one_or_none()

            # Require active account with admin role and valid password
            if not account or not account.is_active or account.role != "admin":
                return False

            if not verify_password(password, account.password_hash):
                return False

            request.session.update({
                "admin_user_id": str(account.id),
                "admin_email": account.email,
            })
            return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool | RedirectResponse:
        admin_user_id = request.session.get("admin_user_id")
        if not admin_user_id:
            return False

        try:
            user_uuid = uuid.UUID(admin_user_id)
        except ValueError:
            request.session.clear()
            return False

        async with get_admin_session_maker()() as db:
            stmt = select(Account).where(Account.id == user_uuid)
            res = await db.execute(stmt)
            account = res.scalar_one_or_none()

            if not account or not account.is_active or account.role != "admin":
                request.session.clear()
                return False

        return True

class AccountAdmin(ModelView, model=Account):
    column_list = ["id", "email", "role", "subscription_plan", "is_active", "created_at"]
    column_searchable_list = ["email", "role", "subscription_plan"]
    icon = "fa-solid fa-users"

class AuthSessionAdmin(ModelView, model=AuthSession):
    column_list = ["id", "owner_id", "device_info", "is_revoked", "expires_at"]
    icon = "fa-solid fa-key"

class RecoveryCodeAdmin(ModelView, model=RecoveryCode):
    column_list = ["id", "owner_id", "is_used", "used_at"]
    icon = "fa-solid fa-shield-halved"

class ProfileSyncAdmin(ModelView, model=ProfileSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-id-card"

class TasksSyncAdmin(ModelView, model=TasksSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at", "deleted_at"]
    column_searchable_list = ["client_id"]
    icon = "fa-solid fa-list-check"

class EventsSyncAdmin(ModelView, model=EventsSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    column_searchable_list = ["client_id"]
    icon = "fa-solid fa-calendar"

class TimeBlocksSyncAdmin(ModelView, model=TimeBlocksSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    column_searchable_list = ["client_id"]
    icon = "fa-solid fa-clock"

class RemindersSyncAdmin(ModelView, model=RemindersSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    column_searchable_list = ["client_id"]
    icon = "fa-solid fa-bell"

class NotesSyncAdmin(ModelView, model=NotesSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    column_searchable_list = ["client_id"]
    icon = "fa-solid fa-sticky-note"

class CustomCategoriesSyncAdmin(ModelView, model=CustomCategoriesSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-tags"

class PomodoroSettingsSyncAdmin(ModelView, model=PomodoroSettingsSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-stopwatch"

class PomodoroSessionsSyncAdmin(ModelView, model=PomodoroSessionsSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-hourglass-half"

class FlashcardDecksSyncAdmin(ModelView, model=FlashcardDecksSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-layer-group"

class StudySummariesSyncAdmin(ModelView, model=StudySummariesSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-book-open"

class RecordedMeetingsSyncAdmin(ModelView, model=RecordedMeetingsSync):
    column_list = ["owner_id", "client_id", "version", "change_id", "updated_at"]
    icon = "fa-solid fa-microphone-lines"

class IdempotentMutationAdmin(ModelView, model=IdempotentMutation):
    column_list = ["mutation_id", "owner_id", "status", "created_at"]
    icon = "fa-solid fa-fingerprint"

class ChangeFeedAdmin(ModelView, model=ChangeFeed):
    column_list = ["change_id", "owner_id", "entity_type", "entity_id", "operation", "created_at"]
    icon = "fa-solid fa-stream"

class SyncHeadAdmin(ModelView, model=SyncHead):
    column_list = ["owner_id", "latest_change_id", "updated_at"]
    icon = "fa-solid fa-arrows-rotate"

class AIUsageAdmin(ModelView, model=AIUsage):
    column_list = ["id", "owner_id", "request_type", "prompt_tokens", "completion_tokens", "created_at"]
    icon = "fa-solid fa-robot"

class FeatureFlagAdmin(ModelView, model=FeatureFlag):
    """Runtime switches. Only `enabled` can be edited; the next request sees it."""
    name = "Feature Flag"
    name_plural = "Feature Flags"
    column_list = ["key", "enabled", "description", "updated_at"]
    column_labels = {"key": "Flag", "enabled": "On"}
    form_columns = ["enabled"]
    # The code looks flags up by key, so they are neither added nor removed here.
    can_create = False
    can_delete = False
    icon = "fa-solid fa-toggle-on"

class SecurityEventAdmin(ModelView, model=SecurityEvent):
    column_list = ["id", "owner_id", "event_type", "ip_address", "created_at"]
    icon = "fa-solid fa-shield-virus"

authentication_backend = AdminAuth(secret_key=settings.JWT_PRIVATE_KEY[:32])

def setup_admin(app, engine):
    admin = Admin(
        app,
        engine,
        authentication_backend=authentication_backend,
        title="LAFINA Studio (Admin UI)",
        base_url="/admin"
    )
    admin.add_view(AccountAdmin)
    admin.add_view(AuthSessionAdmin)
    admin.add_view(RecoveryCodeAdmin)
    admin.add_view(ProfileSyncAdmin)
    admin.add_view(TasksSyncAdmin)
    admin.add_view(EventsSyncAdmin)
    admin.add_view(TimeBlocksSyncAdmin)
    admin.add_view(RemindersSyncAdmin)
    admin.add_view(NotesSyncAdmin)
    admin.add_view(CustomCategoriesSyncAdmin)
    admin.add_view(PomodoroSettingsSyncAdmin)
    admin.add_view(PomodoroSessionsSyncAdmin)
    admin.add_view(FlashcardDecksSyncAdmin)
    admin.add_view(StudySummariesSyncAdmin)
    admin.add_view(RecordedMeetingsSyncAdmin)
    admin.add_view(IdempotentMutationAdmin)
    admin.add_view(ChangeFeedAdmin)
    admin.add_view(SyncHeadAdmin)
    admin.add_view(FeatureFlagAdmin)
    admin.add_view(AIUsageAdmin)
    admin.add_view(SecurityEventAdmin)
    app.state.admin = admin
    return admin
