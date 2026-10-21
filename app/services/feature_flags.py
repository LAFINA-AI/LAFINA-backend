"""Runtime switches, read from the database on each use.

An administrator flips a flag in the admin panel (Feature Flags) and the next
request sees it: no restart, no redeploy. A flag with no row yet falls back
to its default, and startup writes the default rows so they appear in the
panel ready to flip.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feature_flag import FeatureFlag

logger = logging.getLogger("lafina.feature_flags")

HANDBOOK_RAG = "handbook_rag"

DEFAULT_FLAGS: dict[str, tuple[bool, str]] = {
    HANDBOOK_RAG: (
        True,
        "Online assistant answers university questions from the USTP Student Handbook "
        "(Pinecone). Turn off to answer from the model alone.",
    ),
}


async def is_enabled(db: AsyncSession, key: str) -> bool:
    """Whether a flag is on. An unreadable flag is treated as its default."""
    default = DEFAULT_FLAGS.get(key, (False, ""))[0]
    try:
        flag = await db.get(FeatureFlag, key)
    except SQLAlchemyError as exc:
        logger.warning(f"Feature flag {key!r} could not be read ({type(exc).__name__}); using default")
        await db.rollback()
        return default
    return default if flag is None else bool(flag.enabled)


async def ensure_default_flags(db: AsyncSession) -> None:
    """Adds any missing flag rows with their defaults. Existing choices are kept."""
    existing = set((await db.execute(select(FeatureFlag.key))).scalars().all())
    for key, (enabled, description) in DEFAULT_FLAGS.items():
        if key not in existing:
            db.add(FeatureFlag(key=key, enabled=enabled, description=description))
    await db.commit()
