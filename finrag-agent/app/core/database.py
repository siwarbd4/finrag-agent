"""
Database Configuration and Initialization (SQLAlchemy + SQLite)
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class Document(Base):
    """Document metadata model."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    page_count = Column(Integer, nullable=True)
    doc_type = Column(String(100), nullable=True)   # rapport_annuel, prospectus, etc.
    language = Column(String(10), default="fr")
    status = Column(String(50), default="pending")  # pending | indexed | error
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "doc_type": self.doc_type,
            "language": self.language,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class QueryLog(Base):
    """Query history and analytics."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)      # JSON list of source doc IDs
    processing_time_ms = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    chunks_retrieved = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


# Async engine
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Create all tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency for getting DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
