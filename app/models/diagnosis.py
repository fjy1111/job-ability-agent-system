from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class DiagnosisRecord(Base):
    """
    学生历史诊断记录表
    """
    __tablename__ = "diagnosis_records"
    __table_args__ = (
        Index("ix_diagnosis_records_user_resume_hash", "user_id", "resume_hash"),
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    name: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    major: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    grade: Mapped[str] = mapped_column(
        String(30),
        nullable=False
    )

    target_job: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    skills: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    projects: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    competitions: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    certificates: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    self_intro: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    professional_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    practice_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    tools_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    career_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    agent_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending"
    )

    agent_result_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    resume_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
