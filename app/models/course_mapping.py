from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class AbilityTag(Base):
    """
    标准能力标签表。
    课程和岗位统一映射到能力标签，避免直接关键词硬匹配。
    """
    __tablename__ = "ability_tags"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    tag_name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False
    )

    category: Mapped[str] = mapped_column(
        String(50),
        default="通用能力",
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class CourseRecord(Base):
    """
    课程资源表。
    保存课程名称、课程大纲、课程目标等可被智能体抽取能力标签的原始数据。
    """
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    course_code: Mapped[str | None] = mapped_column(
        String(50),
        unique=True,
        nullable=True
    )

    course_name: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False
    )

    course_type: Mapped[str] = mapped_column(
        String(50),
        default="专业课程",
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    syllabus_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class CourseAbilityRelation(Base):
    """
    课程-能力关系表。
    coverage_score 表示课程覆盖度，cultivate_level 表示课程可培养到的能力等级。
    """
    __tablename__ = "course_ability_relations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    course_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("courses.id"),
        nullable=False
    )

    ability_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ability_tags.id"),
        nullable=False
    )

    coverage_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )

    cultivate_level: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )

    evidence_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class JobAbilityRelation(Base):
    """
    岗位-能力关系表。
    required_level 表示岗位要求等级，weight 表示能力对岗位的重要程度。
    """
    __tablename__ = "job_ability_relations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    job_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("job_knowledge_records.id"),
        nullable=False
    )

    ability_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ability_tags.id"),
        nullable=False
    )

    required_level: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )

    weight: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False
    )

    evidence_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class StudentCourseRecord(Base):
    """
    学生课程记录表。
    支持手动填写、成绩单导入或后续教务数据同步。
    """
    __tablename__ = "student_courses"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    diagnosis_record_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("diagnosis_records.id"),
        nullable=True
    )

    course_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("courses.id"),
        nullable=True
    )

    course_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False
    )

    score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    study_status: Mapped[str] = mapped_column(
        String(30),
        default="已学",
        nullable=False
    )

    evidence_text: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    source_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("data_sources.id"),
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class StudentAbilityProfileRecord(Base):
    """
    学生能力画像明细表。
    保存每个能力标签的掌握等级、证据和置信度。
    """
    __tablename__ = "student_ability_profiles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    diagnosis_record_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("diagnosis_records.id"),
        nullable=True
    )

    ability_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ability_tags.id"),
        nullable=False
    )

    current_level: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )

    evidence_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class MappingResultRecord(Base):
    """
    智能映射结果表。
    保存一次课程-能力-岗位映射的结果，便于追踪、复盘和展示。
    """
    __tablename__ = "mapping_results"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    diagnosis_record_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("diagnosis_records.id"),
        nullable=True
    )

    job_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("job_knowledge_records.id"),
        nullable=True
    )

    match_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )

    matched_abilities_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    missing_abilities_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    recommended_courses_json: Mapped[str] = mapped_column(
        Text,
        default="[]"
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
