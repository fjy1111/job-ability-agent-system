from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class DataSourceRecord(Base):
    """
    数据来源表。
    所有真实采集文件、临时模拟培养计划、简历文件都在这里登记来源。
    """
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    source_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )

    source_type: Mapped[str] = mapped_column(
        String(80),
        nullable=False
    )

    source_platform: Mapped[str] = mapped_column(
        String(80),
        default=""
    )

    source_path: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    file_sha256: Mapped[str] = mapped_column(
        String(64),
        default=""
    )

    source_status: Mapped[str] = mapped_column(
        String(50),
        default="raw_collected",
        nullable=False
    )

    is_synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class JobPostRecord(Base):
    """
    招聘岗位原始记录表。
    一行对应一个 Excel 中的岗位样本。
    """
    __tablename__ = "job_post_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=False
    )

    sheet_name: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    row_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    row_hash: Mapped[str] = mapped_column(
        String(64),
        default=""
    )

    job_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )

    company_name: Mapped[str] = mapped_column(
        String(200),
        default=""
    )

    hiring_city: Mapped[str] = mapped_column(
        String(120),
        default=""
    )

    educational_requirements: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    required_skills_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    related_projects_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    recommended_courses_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    recommended_certificates_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    salary_range: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    raw_json: Mapped[str] = mapped_column(
        Text,
        default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class ResumeRecord(Base):
    """
    简历原始文本与抽取结果表。
    """
    __tablename__ = "resume_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    source_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=False
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    applicant_name: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    resume_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    extracted_skills_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    extracted_courses_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    extracted_projects_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    raw_text_hash: Mapped[str] = mapped_column(
        String(64),
        default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class ExtractionLogRecord(Base):
    """
    智能抽取日志表。
    记录抽取方法、目标表、输入哈希、输出 JSON 和复核状态。
    """
    __tablename__ = "extraction_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    target_table: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    target_record_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    extraction_method: Mapped[str] = mapped_column(
        String(100),
        default="manual_import"
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    prompt_version: Mapped[str] = mapped_column(
        String(50),
        default=""
    )

    input_hash: Mapped[str] = mapped_column(
        String(64),
        default=""
    )

    output_json: Mapped[str] = mapped_column(
        Text,
        default="{}"
    )

    review_status: Mapped[str] = mapped_column(
        String(50),
        default="pending_review"
    )

    reviewer: Mapped[str] = mapped_column(
        String(100),
        default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
