from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class JobKnowledgeRecord(Base):
    """
    岗位能力知识图谱表
    用数据库保存岗位与技能、项目、课程、证书之间的关系
    """
    __tablename__ = "job_knowledge_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    job_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    required_skills_json: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    related_projects_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    recommended_courses_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    recommended_certificates_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )