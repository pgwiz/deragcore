"""Database configuration and async session management."""

import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from ragcore.config import settings

logger = logging.getLogger(__name__)

# Create declarative base for all models
Base = declarative_base()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level == "DEBUG",
    future=True,
    pool_size=20,
    max_overflow=10,
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncSession:
    """Get a database session for dependency injection."""
    async with async_session_factory() as session:
        yield session


async def init_db():
    """Initialize database schema."""
    logger.info("Initializing database...")

    # Import all models to register with Base.metadata
    # This is done here to avoid circular imports
    from ragcore.auth import models as auth_models  # noqa: F401
    from ragcore.webhooks import models as webhook_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def close_db():
    """Close database connection pool."""
    logger.info("Closing database...")
    await engine.dispose()
    logger.info("Database closed")
