import json
import os
import hashlib
import logging
import re
import secrets
import subprocess
import tempfile
from html import escape as html_escape
from pathlib import Path
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI, Request, Response, Form, Depends, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    or_,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)

from app.services.llm_ability_match_service import (
    calculate_job_match,
    refine_job_matches_with_llm,
)
from app.services.ability_match_service import (
    match_profile_to_job as match_profile_to_job_local,
    score_four_dimensions as score_four_dimensions_local,
)
from app.agent.diagnosis_agent import (
    AGENT_ROSTER,
    run_diagnosis_agent,
    run_diagnosis_agent_stream,
)
from app.services.llm_errors import LLMCallError
from app.services.llm_gap_path_agent import generate_top5_gap_paths
from app.services.match_cache_service import (
    MATCH_CACHE_ALGORITHM_VERSION,
    build_job_version,
    build_match_profile_version,
)
from app.services.mock_interview_service import (
    build_interview_session,
    build_written_exam,
    grade_written_exam,
    respond_to_interview_answer,
)
from app.services.resume_optimizer_service import (
    extract_resume_text_from_upload,
    optimize_resume,
)
from app.services.course_job_mapping_service import (
    build_course_job_mapping_graph,
    extract_courses_from_resume,
)
from app.services.course_ability_inference_service import (
    COURSE_ABILITY_PROMPT_VERSION,
    infer_course_abilities_with_llm,
)
from app.services.resume_profile_extractor_service import (
    extract_student_profile_from_resume,
)
from app.services.ability_profile_cache_service import (
    attach_resume_cache_metadata,
    build_cached_agent_events,
    build_resume_cache_hash,
    is_matching_cached_result,
)
from app.services.privacy_service import (
    candidate_privacy_label,
    enforce_collaboration_rate_limit,
    env_flag,
    get_or_create_csrf_token,
    hash_password,
    institution_role_access_error,
    redact_sensitive_resume_text,
    require_valid_csrf,
    verify_password,
)

# =========================================================
# 项目路径配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

DATA_DIR = BASE_DIR / "data"
LATEST_STUDENT_FILE = DATA_DIR / "latest_student.json"

# 读取项目根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)


# =========================================================
# 数据库配置
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("没有读取到 DATABASE_URL，请检查项目根目录下是否存在 .env 文件。")

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true",
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


class Base(DeclarativeBase):
    """
    所有数据库表模型的基础类
    """
    pass


# =========================================================
# 数据库表模型
# =========================================================

class User(Base):
    """
    用户表：用于注册和登录。
    新注册密码使用 PBKDF2-SHA256 保存；旧明文账号首次成功登录后自动升级。
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False
    )

    password: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class DiagnosisRecord(Base):
    """
    学生历史诊断记录表
    """
    __tablename__ = "diagnosis_records"
    __table_args__ = (
        Index("ix_diagnosis_records_user_resume_hash", "user_id", "resume_hash"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    major: Mapped[str] = mapped_column(String(100), nullable=False)
    grade: Mapped[str] = mapped_column(String(30), nullable=False)
    target_job: Mapped[str] = mapped_column(String(100), nullable=False)

    skills: Mapped[str] = mapped_column(Text, nullable=False)
    projects: Mapped[str] = mapped_column(Text, default="")
    competitions: Mapped[str] = mapped_column(Text, default="")
    certificates: Mapped[str] = mapped_column(Text, default="")
    self_intro: Mapped[str] = mapped_column(Text, default="")

    professional_score: Mapped[int] = mapped_column(Integer, nullable=False)
    practice_score: Mapped[int] = mapped_column(Integer, nullable=False)
    tools_score: Mapped[int] = mapped_column(Integer, nullable=False)
    career_score: Mapped[int] = mapped_column(Integer, nullable=False)

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


class JobKnowledgeRecord(Base):
    """
    岗位能力知识图谱表
    用数据库保存岗位与技能、项目、课程、证书之间的关系。
    """
    __tablename__ = "job_knowledge_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    job_name: Mapped[str] = mapped_column(String(100), nullable=False)
    company_name: Mapped[str] = mapped_column(String(100), default="")
    hiring_city: Mapped[str] = mapped_column(String(100), default="")
    educational_requirements: Mapped[str] = mapped_column(String(200), default="")
    salary_range: Mapped[str] = mapped_column(String(100), default="")
    required_skills_json: Mapped[str] = mapped_column(Text, nullable=False)
    related_projects_json: Mapped[str] = mapped_column(Text, default="[]")
    recommended_courses_json: Mapped[str] = mapped_column(Text, default="[]")
    recommended_certificates_json: Mapped[str] = mapped_column(Text, default="[]")

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class EmploymentGuidanceRecord(Base):
    """
    精准就业指导生成记录。
    """
    __tablename__ = "employment_guidance_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True
    )
    message: Mapped[str] = mapped_column(Text, default="")
    uploaded_filename: Mapped[str] = mapped_column(String(255), default="")
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class JobMatchCacheRecord(Base):
    """
    岗位匹配持久化缓存。

    同一诊断记录、岗位库版本和算法版本可分别保存本地排序与 LLM 精排结果。
    """
    __tablename__ = "job_match_cache_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    cache_key: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True
    )
    diagnosis_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("diagnosis_records.id"),
        nullable=False,
        index=True
    )
    job_version: Mapped[str] = mapped_column(String(64), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(100), nullable=False)
    result_type: Mapped[str] = mapped_column(String(20), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False
    )


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
    一行对应一个 Excel 中的岗位样本，保留公司、城市、薪资和原始 JSON。
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


class AbilityTag(Base):
    """
    标准能力标签表。
    课程和岗位不直接硬匹配，而是统一映射到能力标签上。
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
    保存课程名称、大纲、目标等可被智能体抽取能力标签的原始数据。
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
    coverage_score 表示课程对能力的覆盖度，cultivate_level 表示课程能培养到的能力等级。
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


class CourseAbilityInferenceRecord(Base):
    """
    AI 推理课程能力待审核表。
    只保存本地课程知识库未命中时的大模型推理结果，不混入真实课程-能力关系表。
    """
    __tablename__ = "course_ability_inference_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    course_name: Mapped[str] = mapped_column(
        String(120),
        index=True,
        nullable=False
    )
    abilities_json: Mapped[str] = mapped_column(
        Text,
        default="[]",
        nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )
    reason: Mapped[str] = mapped_column(
        Text,
        default=""
    )
    prompt_version: Mapped[str] = mapped_column(
        String(60),
        default=COURSE_ABILITY_PROMPT_VERSION,
        nullable=False
    )
    model_name: Mapped[str] = mapped_column(
        String(120),
        default=""
    )
    source_type: Mapped[str] = mapped_column(
        String(50),
        default="llm_inference",
        nullable=False
    )
    source_label: Mapped[str] = mapped_column(
        String(30),
        default="AI推理",
        nullable=False
    )
    review_status: Mapped[str] = mapped_column(
        String(30),
        default="pending_review",
        nullable=False
    )
    raw_response_json: Mapped[str] = mapped_column(
        Text,
        default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )


class JobAbilityRelation(Base):
    """
    岗位-能力关系表。
    required_level 表示岗位要求等级，weight 表示该能力对岗位的重要程度。
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
    后续可以从成绩单导入，也可以由学生手动填写。
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
    保存一次课程-能力-岗位映射的结果，便于追踪、复盘和答辩展示。
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


class ResumeRecord(Base):
    """
    简历原始文本与抽取结果表。
    保存 PDF 抽取文本、技能、课程和项目证据，便于个性化匹配追溯。
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


class TriPartyResumeApplicationRecord(Base):
    """
    学生-学校-企业三方协同投递记录。
    简历先进入学校审核，未发现造假后再流转到企业评估。
    """
    __tablename__ = "triparty_resume_applications"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    student_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True
    )
    student_username: Mapped[str] = mapped_column(
        String(50),
        default=""
    )
    student_name: Mapped[str] = mapped_column(
        String(100),
        default=""
    )
    major: Mapped[str] = mapped_column(
        String(120),
        default=""
    )
    target_company: Mapped[str] = mapped_column(
        String(160),
        default=""
    )
    target_job: Mapped[str] = mapped_column(
        String(160),
        default=""
    )
    resume_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    resume_hash: Mapped[str] = mapped_column(
        String(64),
        default="",
        index=True
    )
    status: Mapped[str] = mapped_column(
        String(40),
        default="school_review",
        nullable=False,
        index=True
    )
    school_reviewer: Mapped[str] = mapped_column(
        String(100),
        default=""
    )
    school_review_result: Mapped[str] = mapped_column(
        String(40),
        default=""
    )
    school_feedback: Mapped[str] = mapped_column(
        Text,
        default=""
    )
    warning_message: Mapped[str] = mapped_column(
        Text,
        default=""
    )
    forwarded_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )
    enterprise_reviewer: Mapped[str] = mapped_column(
        String(100),
        default=""
    )
    enterprise_decision: Mapped[str] = mapped_column(
        String(40),
        default=""
    )
    enterprise_advice_to_student: Mapped[str] = mapped_column(
        Text,
        default=""
    )
    enterprise_advice_to_school: Mapped[str] = mapped_column(
        Text,
        default=""
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False
    )


# =========================================================
# 数据库初始化
# =========================================================

def init_job_knowledge_data():
    """
    初始化岗位能力知识图谱数据。
    如果数据库中已有岗位数据，则不重复插入。
    """
    db = SessionLocal()

    try:
        count = db.query(JobKnowledgeRecord).count()

        if count > 0:
            print("[JobKnowledge] 岗位知识图谱数据已存在，无需重复初始化")
            return

        default_jobs = [
            {
                "job_name": "AI应用开发工程师",
                "required_skills": ["Python", "FastAPI", "LangChain", "SQL", "大模型API调用"],
                "related_projects": ["AI聊天助手", "知识库问答系统", "简历生成系统"],
                "recommended_courses": ["Python程序设计", "数据库原理", "人工智能导论", "大模型应用开发实训"],
                "recommended_certificates": ["英语六级", "软考程序员"]
            },
            {
                "job_name": "Java后端工程师",
                "required_skills": ["Java", "Spring Boot", "MySQL", "Redis", "Linux"],
                "related_projects": ["学生管理系统", "电商后台系统", "权限管理系统"],
                "recommended_courses": ["Java程序设计", "数据库原理", "软件工程", "Linux操作系统"],
                "recommended_certificates": ["软考程序员", "英语六级"]
            },
            {
                "job_name": "数据分析师",
                "required_skills": ["Python", "SQL", "Excel", "Pandas", "数据可视化"],
                "related_projects": ["数据分析项目", "用户画像分析", "销售数据分析"],
                "recommended_courses": ["概率论与数理统计", "数据库原理", "数据挖掘"],
                "recommended_certificates": ["英语六级"]
            },
            {
                "job_name": "前端开发工程师",
                "required_skills": ["HTML", "CSS", "JavaScript", "Vue", "接口调用"],
                "related_projects": ["个人博客", "后台管理系统", "数据可视化大屏"],
                "recommended_courses": ["Web前端开发", "软件工程"],
                "recommended_certificates": ["英语六级"]
            },
            {
                "job_name": "测试工程师",
                "required_skills": ["软件测试", "Python", "接口测试", "Linux", "自动化测试"],
                "related_projects": ["接口测试项目", "自动化测试脚本", "缺陷管理系统"],
                "recommended_courses": ["软件工程", "软件测试技术"],
                "recommended_certificates": ["软件测试相关证书"]
            }
        ]

        for job in default_jobs:
            record = JobKnowledgeRecord(
                job_name=job["job_name"],
                required_skills_json=json.dumps(
                    job["required_skills"],
                    ensure_ascii=False
                ),
                related_projects_json=json.dumps(
                    job["related_projects"],
                    ensure_ascii=False
                ),
                recommended_courses_json=json.dumps(
                    job["recommended_courses"],
                    ensure_ascii=False
                ),
                recommended_certificates_json=json.dumps(
                    job["recommended_certificates"],
                    ensure_ascii=False
                )
            )

            db.add(record)

        db.commit()
        print("[JobKnowledge] 岗位知识图谱初始化完成")

    finally:
        db.close()


def init_course_job_mapping_data():
    """
    初始化课程-能力-岗位智能映射底座。
    这部分数据全部写入 MySQL，后续可由大模型抽取结果持续更新。
    """
    db = SessionLocal()

    try:
        ability_specs = [
            {"tag_name": "Python", "category": "编程语言", "description": "使用 Python 完成脚本、后端、数据处理或 AI 应用开发。"},
            {"tag_name": "Java", "category": "编程语言", "description": "使用 Java 完成面向对象编程和企业级后端开发。"},
            {"tag_name": "SQL", "category": "数据库能力", "description": "使用 SQL 完成数据查询、聚合、关联和统计分析。"},
            {"tag_name": "MySQL", "category": "数据库能力", "description": "使用 MySQL 完成业务数据存储、查询和优化。"},
            {"tag_name": "数据库设计", "category": "数据库能力", "description": "进行数据建模、表结构设计、索引设计和事务理解。"},
            {"tag_name": "数据结构", "category": "计算机基础", "description": "理解数组、链表、树、图、栈、队列等基础数据结构。"},
            {"tag_name": "算法", "category": "计算机基础", "description": "掌握常见算法思想、复杂度分析和基础编码能力。"},
            {"tag_name": "软件工程", "category": "工程能力", "description": "理解需求分析、系统设计、编码规范、测试和文档协作。"},
            {"tag_name": "需求分析", "category": "工程能力", "description": "将业务需求拆解为功能需求和系统约束。"},
            {"tag_name": "项目文档", "category": "工程能力", "description": "编写 README、接口文档、设计说明和项目总结。"},
            {"tag_name": "FastAPI", "category": "后端开发", "description": "使用 FastAPI 构建后端接口和 AI 应用服务。"},
            {"tag_name": "Spring Boot", "category": "后端开发", "description": "使用 Spring Boot 构建企业级 Java 后端服务。"},
            {"tag_name": "Linux", "category": "工程工具", "description": "掌握 Linux 常用命令、环境部署和服务运维基础。"},
            {"tag_name": "Redis", "category": "后端开发", "description": "理解缓存、键值存储和常见 Redis 使用场景。"},
            {"tag_name": "HTML", "category": "前端开发", "description": "使用 HTML 构建页面结构。"},
            {"tag_name": "CSS", "category": "前端开发", "description": "使用 CSS 完成页面布局和视觉样式。"},
            {"tag_name": "JavaScript", "category": "前端开发", "description": "使用 JavaScript 完成交互逻辑和接口调用。"},
            {"tag_name": "Vue", "category": "前端开发", "description": "使用 Vue 构建组件化前端应用。"},
            {"tag_name": "接口调用", "category": "工程能力", "description": "理解 HTTP 接口、请求参数、响应处理和前后端协作。"},
            {"tag_name": "软件测试", "category": "测试能力", "description": "理解测试用例、缺陷管理、功能测试和质量保障流程。"},
            {"tag_name": "接口测试", "category": "测试能力", "description": "围绕 API 接口进行参数校验、响应校验和自动化测试。"},
            {"tag_name": "自动化测试", "category": "测试能力", "description": "使用脚本或测试框架提升测试执行效率。"},
            {"tag_name": "数据分析", "category": "数据能力", "description": "围绕业务问题进行数据清洗、统计分析和结论表达。"},
            {"tag_name": "Pandas", "category": "数据能力", "description": "使用 Pandas 完成数据读取、清洗、转换和分析。"},
            {"tag_name": "Excel", "category": "数据能力", "description": "使用 Excel 完成基础数据处理、透视分析和图表展示。"},
            {"tag_name": "数据可视化", "category": "数据能力", "description": "将数据分析结论以图表或仪表盘方式呈现。"},
            {"tag_name": "统计学", "category": "数学基础", "description": "理解概率、统计推断、分布和基础建模思想。"},
            {"tag_name": "机器学习", "category": "人工智能", "description": "理解监督学习、无监督学习和常见模型训练流程。"},
            {"tag_name": "人工智能基础", "category": "人工智能", "description": "理解 AI 基础概念、典型任务和应用场景。"},
            {"tag_name": "大模型API调用", "category": "人工智能", "description": "调用大模型 API 完成问答、生成、抽取和智能体任务。"},
            {"tag_name": "LangChain", "category": "人工智能", "description": "使用 LangChain 编排大模型应用链路。"},
            {"tag_name": "RAG知识库", "category": "人工智能", "description": "构建检索增强生成知识库问答系统。"}
        ]

        course_specs = [
            {
                "course_code": "CS101",
                "course_name": "Python程序设计",
                "course_type": "专业基础课",
                "description": "面向程序设计基础、脚本开发和数据处理的课程。",
                "syllabus_text": "Python语法、函数、文件处理、异常、面向对象、基础项目实践。",
                "abilities": [
                    {"tag_name": "Python", "coverage_score": 0.9, "cultivate_level": 2},
                    {"tag_name": "算法", "coverage_score": 0.4, "cultivate_level": 1}
                ]
            },
            {
                "course_code": "CS102",
                "course_name": "数据结构与算法",
                "course_type": "专业核心课",
                "description": "训练基础数据结构、算法思维和编码能力。",
                "syllabus_text": "线性表、栈、队列、树、图、排序、查找、复杂度分析。",
                "abilities": [
                    {"tag_name": "数据结构", "coverage_score": 0.9, "cultivate_level": 3},
                    {"tag_name": "算法", "coverage_score": 0.8, "cultivate_level": 3}
                ]
            },
            {
                "course_code": "CS201",
                "course_name": "数据库原理",
                "course_type": "专业核心课",
                "description": "面向关系数据库、SQL 和数据库设计的课程。",
                "syllabus_text": "关系模型、SQL、索引、事务、范式、数据库设计、MySQL实践。",
                "abilities": [
                    {"tag_name": "SQL", "coverage_score": 0.9, "cultivate_level": 3},
                    {"tag_name": "MySQL", "coverage_score": 0.75, "cultivate_level": 2},
                    {"tag_name": "数据库设计", "coverage_score": 0.8, "cultivate_level": 3}
                ]
            },
            {
                "course_code": "CS202",
                "course_name": "软件工程",
                "course_type": "专业核心课",
                "description": "覆盖软件开发流程、需求分析、设计、测试和文档协作。",
                "syllabus_text": "需求分析、系统设计、编码规范、测试、项目管理、文档编写。",
                "abilities": [
                    {"tag_name": "软件工程", "coverage_score": 0.9, "cultivate_level": 3},
                    {"tag_name": "需求分析", "coverage_score": 0.75, "cultivate_level": 2},
                    {"tag_name": "项目文档", "coverage_score": 0.7, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "CS203",
                "course_name": "Java程序设计",
                "course_type": "专业基础课",
                "description": "面向 Java 语法、面向对象和后端开发基础的课程。",
                "syllabus_text": "Java基础语法、集合、异常、面向对象、JDBC、基础后端项目。",
                "abilities": [
                    {"tag_name": "Java", "coverage_score": 0.9, "cultivate_level": 2},
                    {"tag_name": "MySQL", "coverage_score": 0.45, "cultivate_level": 1}
                ]
            },
            {
                "course_code": "CS204",
                "course_name": "Web前端开发",
                "course_type": "专业方向课",
                "description": "面向网页开发、前端交互和接口调用的课程。",
                "syllabus_text": "HTML、CSS、JavaScript、Vue、组件开发、接口调用。",
                "abilities": [
                    {"tag_name": "HTML", "coverage_score": 0.85, "cultivate_level": 2},
                    {"tag_name": "CSS", "coverage_score": 0.85, "cultivate_level": 2},
                    {"tag_name": "JavaScript", "coverage_score": 0.9, "cultivate_level": 2},
                    {"tag_name": "Vue", "coverage_score": 0.75, "cultivate_level": 2},
                    {"tag_name": "接口调用", "coverage_score": 0.7, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "CS205",
                "course_name": "Linux操作系统",
                "course_type": "专业方向课",
                "description": "面向 Linux 命令、部署环境和服务器运维基础的课程。",
                "syllabus_text": "Linux文件系统、Shell命令、权限、进程、服务部署、日志排查。",
                "abilities": [
                    {"tag_name": "Linux", "coverage_score": 0.9, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "AI101",
                "course_name": "人工智能导论",
                "course_type": "专业方向课",
                "description": "介绍人工智能基础概念、机器学习方法和典型应用场景。",
                "syllabus_text": "人工智能概念、搜索、机器学习、神经网络、自然语言处理、AI应用案例。",
                "abilities": [
                    {"tag_name": "人工智能基础", "coverage_score": 0.9, "cultivate_level": 2},
                    {"tag_name": "机器学习", "coverage_score": 0.6, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "AI201",
                "course_name": "大模型应用开发实训",
                "course_type": "实践课程",
                "description": "面向大模型 API、智能体和知识库问答应用开发的实训课程。",
                "syllabus_text": "大模型API调用、Prompt设计、LangChain、RAG知识库、FastAPI服务封装、项目部署。",
                "abilities": [
                    {"tag_name": "大模型API调用", "coverage_score": 0.9, "cultivate_level": 3},
                    {"tag_name": "LangChain", "coverage_score": 0.85, "cultivate_level": 3},
                    {"tag_name": "RAG知识库", "coverage_score": 0.85, "cultivate_level": 3},
                    {"tag_name": "FastAPI", "coverage_score": 0.65, "cultivate_level": 2},
                    {"tag_name": "Python", "coverage_score": 0.6, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "DA101",
                "course_name": "概率论与数理统计",
                "course_type": "数学基础课",
                "description": "面向概率、统计推断和数据分析基础的课程。",
                "syllabus_text": "概率分布、参数估计、假设检验、回归基础、统计分析方法。",
                "abilities": [
                    {"tag_name": "统计学", "coverage_score": 0.9, "cultivate_level": 3}
                ]
            },
            {
                "course_code": "DA201",
                "course_name": "数据挖掘",
                "course_type": "专业方向课",
                "description": "面向数据处理、建模分析和可视化表达的课程。",
                "syllabus_text": "数据清洗、特征工程、分类聚类、关联分析、Pandas、可视化。",
                "abilities": [
                    {"tag_name": "数据分析", "coverage_score": 0.85, "cultivate_level": 3},
                    {"tag_name": "Pandas", "coverage_score": 0.75, "cultivate_level": 2},
                    {"tag_name": "机器学习", "coverage_score": 0.65, "cultivate_level": 2},
                    {"tag_name": "数据可视化", "coverage_score": 0.7, "cultivate_level": 2}
                ]
            },
            {
                "course_code": "QA101",
                "course_name": "软件测试技术",
                "course_type": "专业方向课",
                "description": "面向测试流程、测试用例和自动化测试基础的课程。",
                "syllabus_text": "测试理论、测试用例设计、缺陷管理、接口测试、自动化测试脚本。",
                "abilities": [
                    {"tag_name": "软件测试", "coverage_score": 0.9, "cultivate_level": 3},
                    {"tag_name": "接口测试", "coverage_score": 0.75, "cultivate_level": 2},
                    {"tag_name": "自动化测试", "coverage_score": 0.7, "cultivate_level": 2}
                ]
            }
        ]

        job_ability_specs = {
            "AI应用开发工程师": [
                {"tag_name": "Python", "required_level": 3, "weight": 0.9},
                {"tag_name": "FastAPI", "required_level": 2, "weight": 0.7},
                {"tag_name": "SQL", "required_level": 2, "weight": 0.6},
                {"tag_name": "大模型API调用", "required_level": 3, "weight": 0.9},
                {"tag_name": "LangChain", "required_level": 2, "weight": 0.75},
                {"tag_name": "RAG知识库", "required_level": 2, "weight": 0.75}
            ],
            "Java后端工程师": [
                {"tag_name": "Java", "required_level": 3, "weight": 0.9},
                {"tag_name": "Spring Boot", "required_level": 3, "weight": 0.9},
                {"tag_name": "MySQL", "required_level": 2, "weight": 0.75},
                {"tag_name": "Redis", "required_level": 2, "weight": 0.6},
                {"tag_name": "Linux", "required_level": 2, "weight": 0.6},
                {"tag_name": "软件工程", "required_level": 2, "weight": 0.55}
            ],
            "数据分析师": [
                {"tag_name": "Python", "required_level": 2, "weight": 0.7},
                {"tag_name": "SQL", "required_level": 3, "weight": 0.85},
                {"tag_name": "Excel", "required_level": 2, "weight": 0.6},
                {"tag_name": "Pandas", "required_level": 2, "weight": 0.75},
                {"tag_name": "数据分析", "required_level": 3, "weight": 0.9},
                {"tag_name": "数据可视化", "required_level": 2, "weight": 0.7},
                {"tag_name": "统计学", "required_level": 2, "weight": 0.65}
            ],
            "前端开发工程师": [
                {"tag_name": "HTML", "required_level": 2, "weight": 0.65},
                {"tag_name": "CSS", "required_level": 2, "weight": 0.65},
                {"tag_name": "JavaScript", "required_level": 3, "weight": 0.9},
                {"tag_name": "Vue", "required_level": 2, "weight": 0.75},
                {"tag_name": "接口调用", "required_level": 2, "weight": 0.7},
                {"tag_name": "项目文档", "required_level": 1, "weight": 0.35}
            ],
            "测试工程师": [
                {"tag_name": "软件测试", "required_level": 3, "weight": 0.9},
                {"tag_name": "接口测试", "required_level": 2, "weight": 0.75},
                {"tag_name": "自动化测试", "required_level": 2, "weight": 0.75},
                {"tag_name": "Python", "required_level": 2, "weight": 0.6},
                {"tag_name": "Linux", "required_level": 1, "weight": 0.45}
            ]
        }

        ability_map = {}
        for spec in ability_specs:
            ability = (
                db.query(AbilityTag)
                .filter(AbilityTag.tag_name == spec["tag_name"])
                .first()
            )
            if ability is None:
                ability = AbilityTag(**spec)
                db.add(ability)
                db.flush()
            ability_map[spec["tag_name"]] = ability

        course_map = {}
        for spec in course_specs:
            abilities = spec.pop("abilities")
            course = (
                db.query(CourseRecord)
                .filter(CourseRecord.course_name == spec["course_name"])
                .first()
            )
            if course is None:
                course = CourseRecord(**spec)
                db.add(course)
                db.flush()
            course_map[course.course_name] = course

            for relation_spec in abilities:
                ability = ability_map.get(relation_spec["tag_name"])
                if ability is None:
                    continue

                existed_relation = (
                    db.query(CourseAbilityRelation)
                    .filter(
                        CourseAbilityRelation.course_id == course.id,
                        CourseAbilityRelation.ability_id == ability.id
                    )
                    .first()
                )
                if existed_relation is None:
                    db.add(
                        CourseAbilityRelation(
                            course_id=course.id,
                            ability_id=ability.id,
                            coverage_score=relation_spec["coverage_score"],
                            cultivate_level=relation_spec["cultivate_level"],
                            evidence_text=f"{course.course_name}覆盖{ability.tag_name}能力"
                        )
                    )

        for job_name, relation_specs in job_ability_specs.items():
            job = (
                db.query(JobKnowledgeRecord)
                .filter(JobKnowledgeRecord.job_name == job_name)
                .first()
            )
            if job is None:
                continue

            for relation_spec in relation_specs:
                ability = ability_map.get(relation_spec["tag_name"])
                if ability is None:
                    continue

                existed_relation = (
                    db.query(JobAbilityRelation)
                    .filter(
                        JobAbilityRelation.job_id == job.id,
                        JobAbilityRelation.ability_id == ability.id
                    )
                    .first()
                )
                if existed_relation is None:
                    db.add(
                        JobAbilityRelation(
                            job_id=job.id,
                            ability_id=ability.id,
                            required_level=relation_spec["required_level"],
                            weight=relation_spec["weight"],
                            evidence_text=f"{job.job_name}岗位要求{ability.tag_name}能力"
                        )
                    )

        db.commit()
        print("[CourseJobMapping] 课程-能力-岗位映射数据初始化完成")

    finally:
        db.close()


def ensure_diagnosis_resume_hash_schema() -> None:
    """为已有 diagnosis_records 平滑补充简历缓存键和联合索引。"""
    inspector = inspect(engine)
    column_names = {
        column["name"]
        for column in inspector.get_columns("diagnosis_records")
    }
    if "resume_hash" not in column_names:
        with engine.begin() as connection:
            connection.execute(text(
                "ALTER TABLE diagnosis_records "
                "ADD COLUMN resume_hash VARCHAR(64) NULL"
            ))

    inspector = inspect(engine)
    index_names = {
        index["name"]
        for index in inspector.get_indexes("diagnosis_records")
    }
    index_name = "ix_diagnosis_records_user_resume_hash"
    if index_name not in index_names:
        with engine.begin() as connection:
            connection.execute(text(
                f"CREATE INDEX {index_name} "
                "ON diagnosis_records (user_id, resume_hash)"
            ))


# 自动创建数据库表，并为旧数据库补充缓存字段。
Base.metadata.create_all(bind=engine)
ensure_diagnosis_resume_hash_schema()

# 默认不初始化演示岗位数据，避免影响真实可追溯数据。
if os.getenv("INIT_DEMO_JOB_DATA", "false").lower() == "true":
    init_job_knowledge_data()
else:
    print("[JobKnowledge] 已跳过演示岗位数据初始化，使用真实采集数据。")

# 默认不初始化演示映射数据，避免影响真实可追溯数据。
if os.getenv("INIT_DEMO_MAPPING_DATA", "false").lower() == "true":
    init_course_job_mapping_data()
else:
    print("[CourseJobMapping] 已跳过演示数据初始化，使用真实采集数据。")


def get_db():
    """
    每次请求创建一个数据库会话，请求完成后自动关闭。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================================================
# FastAPI 应用配置
# =========================================================

app = FastAPI(
    title="岗位能力达成学生成长诊断与精准就业智能体系统",
    description="面向学生成长诊断、岗位匹配和个性化路径规划的智能体系统",
    version="0.1.0"
)

# 未配置固定密钥时使用不可预测的临时密钥，避免继续使用公开默认值。
# 生产环境必须在 .env 配置 SESSION_SECRET_KEY，否则重启后现有登录会话失效。
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "").strip()
if not SESSION_SECRET_KEY:
    SESSION_SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning("SESSION_SECRET_KEY 未配置，已使用本次进程临时密钥。")

SESSION_HTTPS_ONLY = env_flag("SESSION_HTTPS_ONLY", False)
ENFORCE_HTTPS = env_flag("ENFORCE_HTTPS", False)

# Session 中间件：严格同站 Cookie，生产环境可通过 SESSION_HTTPS_ONLY 只允许 HTTPS 传输。
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="strict",
    https_only=SESSION_HTTPS_ONLY,
    max_age=60 * 60 * 2,
)


@app.middleware("http")
async def add_privacy_security_headers(request: Request, call_next):
    """保护登录和三方协同页面，降低缓存泄露、嵌入劫持与被索引风险。"""
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    is_https = request.url.scheme == "https" or forwarded_proto == "https"
    if ENFORCE_HTTPS and not is_https:
        return RedirectResponse(str(request.url.replace(scheme="https")), status_code=307)

    response = await call_next(request)
    protected_path = request.url.path.startswith(("/collaboration", "/login", "/register"))
    if protected_path:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; img-src 'self' data:; "
            "script-src 'self'; form-action 'self'; frame-ancestors 'none'; "
            "base-uri 'self'; object-src 'none'"
        )
    if is_https:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    return PlainTextResponse(
        "User-agent: *\n"
        "Disallow: /collaboration\n"
        "Disallow: /ability\n"
        "Disallow: /job/match\n"
        "Disallow: /history\n"
        "Disallow: /login\n"
        "Disallow: /register\n"
    )


TRIPARTY_ROLE_OPTIONS = (
    {
        "value": "student",
        "label": "学生",
        "description": "投递简历，接收学校审核与企业反馈"
    },
    {
        "value": "school",
        "label": "学校",
        "description": "审核简历真实性，决定是否转交企业"
    },
    {
        "value": "enterprise",
        "label": "企业",
        "description": "评估候选人，给出录用结论和改进建议"
    },
)
TRIPARTY_ROLE_VALUES = {item["value"] for item in TRIPARTY_ROLE_OPTIONS}
TRIPARTY_ROLE_LABELS = {
    item["value"]: item["label"]
    for item in TRIPARTY_ROLE_OPTIONS
}
TRIPARTY_STATUS_LABELS = {
    "school_review": "学校审核中",
    "warning_to_student": "学校退回预警",
    "enterprise_review": "企业评估中",
    "hired": "企业已录用",
    "rejected": "企业已拒录",
}
TRIPARTY_DECISION_LABELS = {
    "hire": "录用",
    "reject": "拒录",
}

ALLOW_TRIPARTY_ROLE_SWITCH = env_flag("ALLOW_TRIPARTY_ROLE_SWITCH", False)


# =========================================================
# 登录状态工具函数
# =========================================================

def normalize_triparty_role(role: str | None) -> str:
    role = (role or "").strip()
    if role in TRIPARTY_ROLE_VALUES:
        return role
    return "student"


def get_session_role(request: Request) -> str:
    return normalize_triparty_role(request.session.get("user_role"))


def get_session_role_label(request: Request) -> str:
    return TRIPARTY_ROLE_LABELS[get_session_role(request)]


def build_login_context(
    title: str = "用户登录",
    error: str = "",
    message: str = "",
    selected_role: str | None = None,
) -> dict:
    return {
        "title": title,
        "error": error,
        "message": message,
        "role_options": TRIPARTY_ROLE_OPTIONS,
        "selected_role": normalize_triparty_role(selected_role),
        "institution_code_required": True,
    }


def get_login_redirect(request: Request):
    """
    检查用户是否已登录。
    未登录返回跳转到 /login 的响应。
    已登录返回 None。
    """
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    return None


def build_collaboration_redirect(
    message: str = "",
    error: str = "",
) -> RedirectResponse:
    params = {}
    if message:
        params["message"] = message
    if error:
        params["error"] = error
    query = f"?{urlencode(params)}" if params else ""
    return RedirectResponse(
        url=f"/collaboration{query}",
        status_code=303
    )


def ensure_triparty_role(
    request: Request,
    expected_role: str,
) -> RedirectResponse | None:
    current_role = get_session_role(request)
    if current_role == expected_role:
        return None

    expected_label = TRIPARTY_ROLE_LABELS.get(expected_role, expected_role)
    return build_collaboration_redirect(
        error=f"请先以{expected_label}身份登录或切换身份后再操作。"
    )


def build_collaboration_context(
    request: Request,
    db: Session,
    message: str = "",
    error: str = "",
    input_data: dict | None = None,
) -> dict:
    role = get_session_role(request)
    user_id = request.session.get("user_id")

    if role == "student":
        applications = (
            db.query(TriPartyResumeApplicationRecord)
            .filter(TriPartyResumeApplicationRecord.student_user_id == user_id)
            .order_by(
                TriPartyResumeApplicationRecord.created_at.desc(),
                TriPartyResumeApplicationRecord.id.desc()
            )
            .all()
        )
    elif role == "school":
        applications = (
            db.query(TriPartyResumeApplicationRecord)
            .order_by(
                (TriPartyResumeApplicationRecord.status == "school_review").desc(),
                TriPartyResumeApplicationRecord.created_at.desc(),
                TriPartyResumeApplicationRecord.id.desc()
            )
            .all()
        )
    else:
        applications = (
            db.query(TriPartyResumeApplicationRecord)
            .filter(
                TriPartyResumeApplicationRecord.status.in_(
                    ["enterprise_review", "hired", "rejected"]
                )
            )
            .order_by(
                (TriPartyResumeApplicationRecord.status == "enterprise_review").desc(),
                TriPartyResumeApplicationRecord.forwarded_at.desc(),
                TriPartyResumeApplicationRecord.created_at.desc(),
                TriPartyResumeApplicationRecord.id.desc()
            )
            .all()
        )

    counts = {
        "school_review": 0,
        "warning_to_student": 0,
        "enterprise_review": 0,
        "hired": 0,
        "rejected": 0,
    }
    # 统计范围也遵循最小权限：学生只看自己的，企业只看已转交企业的记录。
    for record in applications:
        if record.status in counts:
            counts[record.status] += 1

    return {
        "title": "三方数据协同工作台",
        "username": request.session.get("username"),
        "role": role,
        "role_label": TRIPARTY_ROLE_LABELS[role],
        "role_options": TRIPARTY_ROLE_OPTIONS,
        "allow_role_switch": ALLOW_TRIPARTY_ROLE_SWITCH,
        "csrf_token": get_or_create_csrf_token(request),
        "candidate_privacy_label": candidate_privacy_label,
        "status_labels": TRIPARTY_STATUS_LABELS,
        "decision_labels": TRIPARTY_DECISION_LABELS,
        "applications": applications,
        "counts": counts,
        "message": message,
        "error": error,
        "input_data": input_data or {},
    }


# =========================================================
# 业务辅助函数
# =========================================================

def _load_inferred_abilities_from_record(
    record: CourseAbilityInferenceRecord,
) -> dict:
    try:
        abilities = json.loads(record.abilities_json or "[]")
    except json.JSONDecodeError:
        abilities = []
    if not isinstance(abilities, list):
        abilities = []

    return {
        "abilities": [str(item).strip() for item in abilities if str(item).strip()],
        "confidence": record.confidence_score,
        "reason": record.reason,
        "source_label": record.source_label,
        "source_type": record.source_type,
        "review_status": record.review_status,
    }


def _find_existing_course_ability_inference(
    db: Session,
    course_name: str,
) -> CourseAbilityInferenceRecord | None:
    for status in ("accepted", "pending_review"):
        record = (
            db.query(CourseAbilityInferenceRecord)
            .filter(CourseAbilityInferenceRecord.course_name == course_name)
            .filter(CourseAbilityInferenceRecord.review_status == status)
            .order_by(CourseAbilityInferenceRecord.created_at.desc())
            .first()
        )
        if record:
            return record
    return None


def build_course_inferred_ability_map(
    resume_text: str,
    db: Session,
    user_id: int | None = None,
) -> tuple[dict[str, dict], list[str]]:
    inferred_ability_map: dict[str, dict] = {}
    warnings: list[str] = []
    courses = extract_courses_from_resume(resume_text)
    try:
        max_inferences = int(os.getenv("MAX_COURSE_AI_INFERENCES_PER_REQUEST", "5"))
    except ValueError:
        max_inferences = 5
    max_inferences = max(1, max_inferences)
    inference_count = 0

    for course in courses:
        course_name = str(course.get("course_name", "")).strip()
        if not course_name or course.get("abilities"):
            continue

        existing_record = _find_existing_course_ability_inference(db, course_name)
        if existing_record:
            payload = _load_inferred_abilities_from_record(existing_record)
            if payload["abilities"]:
                inferred_ability_map[course_name] = payload
                warnings.append(f"{course_name} 使用已保存的 AI 推理能力标签，状态：{existing_record.review_status}。")
                continue

        if inference_count >= max_inferences:
            warnings.append(f"{course_name} 未在本地课程库命中，本次已达到 AI 推理上限，暂未生成能力标签。")
            continue

        try:
            inference = infer_course_abilities_with_llm(
                course_name=course_name,
                resume_text=resume_text,
            )
        except LLMCallError:
            warnings.append(f"{course_name} 未在本地课程库命中，AI 推理失败，请检查 LLM 配置后重试。")
            continue

        record = CourseAbilityInferenceRecord(
            user_id=user_id,
            course_name=course_name,
            abilities_json=json.dumps(inference["abilities"], ensure_ascii=False),
            confidence_score=float(inference.get("confidence", 0.0)),
            reason=str(inference.get("reason", "")),
            prompt_version=str(inference.get("prompt_version", COURSE_ABILITY_PROMPT_VERSION)),
            model_name=str(inference.get("model_name", "")),
            source_type="llm_inference",
            source_label="AI推理",
            review_status="pending_review",
            raw_response_json=json.dumps(inference, ensure_ascii=False),
        )
        db.add(record)
        db.commit()

        inference_count += 1
        inferred_ability_map[course_name] = inference
        warnings.append(f"{course_name} 未在本地课程库命中，已由大模型推理能力标签并写入待审核表。")

    return inferred_ability_map, warnings


def build_student_data(record: DiagnosisRecord) -> dict:
    """
    将数据库记录转换为模板页面需要的学生信息字典。
    """
    return {
        "name": record.name,
        "major": record.major,
        "grade": record.grade,
        "target_job": record.target_job,
        "skills": record.skills,
        "projects": record.projects,
        "competitions": record.competitions,
        "certificates": record.certificates,
        "self_intro": record.self_intro
    }


def build_ability_scores(record: DiagnosisRecord) -> dict:
    """
    将数据库中的分数字段转换为页面使用的格式。
    """
    return {
        "professional": record.professional_score,
        "practice": record.practice_score,
        "tools": record.tools_score,
        "career": record.career_score
    }


def load_agent_result(record: DiagnosisRecord | None) -> dict:
    """
    从数据库记录中读取智能体诊断结果。
    """
    if record is None or not record.agent_result_json:
        return {}

    try:
        return json.loads(record.agent_result_json)
    except json.JSONDecodeError:
        return {}


def build_match_assessment(record: DiagnosisRecord) -> dict:
    """复用诊断时已经写入数据库的四维画像分数。"""
    agent_result = load_agent_result(record)
    evidence = agent_result.get("score_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "ability_scores": build_ability_scores(record),
        "score_evidence": evidence,
        "recognized_skills": agent_result.get("recognized_skills", []),
        "target_roles": agent_result.get("target_roles", []),
    }


def build_persistent_match_cache_key(
    student_record: DiagnosisRecord,
    job_version: str,
    result_type: str,
) -> str:
    student_data = build_student_data(student_record)
    assessment = build_match_assessment(student_record)
    profile_version = build_match_profile_version(
        student_data,
        assessment,
        user_id=student_record.user_id,
        resume_hash=student_record.resume_hash or "",
    )
    model_name = (
        os.getenv("ABILITY_MATCH_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )
    payload = {
        "profile_version": profile_version,
        "job_version": job_version,
        "model_name": model_name if result_type == "llm" else "local",
        "algorithm_version": MATCH_CACHE_ALGORITHM_VERSION,
        "result_type": result_type,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_persistent_match_cache(
    db: Session,
    student_record: DiagnosisRecord,
    job_version: str,
    result_type: str,
) -> list[dict] | None:
    cache_key = build_persistent_match_cache_key(
        student_record,
        job_version,
        result_type,
    )
    record = (
        db.query(JobMatchCacheRecord)
        .filter(JobMatchCacheRecord.cache_key == cache_key)
        .first()
    )
    if record is None:
        return None
    try:
        value = json.loads(record.result_json)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def save_persistent_match_cache(
    db: Session,
    student_record: DiagnosisRecord,
    job_version: str,
    result_type: str,
    results: list[dict],
) -> None:
    cache_key = build_persistent_match_cache_key(
        student_record,
        job_version,
        result_type,
    )
    (
        db.query(JobMatchCacheRecord)
        .filter(
            JobMatchCacheRecord.diagnosis_id == student_record.id,
            JobMatchCacheRecord.result_type == result_type,
            JobMatchCacheRecord.cache_key != cache_key,
        )
        .delete(synchronize_session=False)
    )
    record = (
        db.query(JobMatchCacheRecord)
        .filter(JobMatchCacheRecord.cache_key == cache_key)
        .first()
    )
    if record is None:
        record = JobMatchCacheRecord(
            cache_key=cache_key,
            diagnosis_id=student_record.id,
            job_version=job_version,
            algorithm_version=MATCH_CACHE_ALGORITHM_VERSION,
            result_type=result_type,
            result_json="[]",
        )
    record.result_json = json.dumps(results, ensure_ascii=False, default=str)
    record.diagnosis_id = student_record.id
    record.job_version = job_version
    record.algorithm_version = MATCH_CACHE_ALGORITHM_VERSION
    record.updated_at = datetime.now()
    db.add(record)
    db.commit()


def attach_cached_gap_paths(
    student_record: DiagnosisRecord,
    job_matches: list[dict],
) -> list[dict]:
    """只挂载已经生成过的路径，不在页面入口调用 LLM。"""
    agent_result = load_agent_result(student_record)
    top5_gap_paths = agent_result.get("top5_gap_paths", [])
    if isinstance(top5_gap_paths, dict):
        top5_gap_paths = top5_gap_paths.get("top5_gap_paths", [])
    if not isinstance(top5_gap_paths, list):
        top5_gap_paths = []

    gap_map = {
        item.get("job_name"): item
        for item in top5_gap_paths
        if isinstance(item, dict) and item.get("job_name")
    }
    for job in job_matches:
        job["gap_detail"] = gap_map.get(job.get("job_name")) or {}
    return job_matches


def get_desktop_dir() -> Path:
    """返回当前 Windows 用户桌面目录；不存在时创建 Desktop 目录。"""
    candidates: list[Path] = []
    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        candidates.extend([
            Path(user_profile) / "Desktop",
            Path(user_profile) / "桌面",
        ])
    candidates.extend([
        Path.home() / "Desktop",
        Path.home() / "桌面",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = candidates[0] if candidates else Path.cwd()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def safe_filename_part(value: str | None, default: str) -> str:
    text = str(value or default).strip() or default
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:48].strip("._ ") or default


def build_export_redirect(path: str, message: str = "", error: str = "") -> RedirectResponse:
    query = urlencode({"export_error": error} if error else {"export_message": message})
    separator = "&" if "?" in path else "?"
    return RedirectResponse(f"{path}{separator}{query}", status_code=303)


def get_job_matches_for_record(
    db: Session,
    student_record: DiagnosisRecord,
    top_n: int = 10,
) -> tuple[list[dict], str, bool]:
    """
    复用 TOP5 页面同一套岗位匹配流程：优先读 AI 精排缓存，其次本地缓存，
    最后调用项目中的岗位匹配算法并写入缓存。
    """
    student_data = build_student_data(student_record)
    assessment = build_match_assessment(student_record)
    job_records = db.query(JobKnowledgeRecord).all()
    job_version = build_job_version(job_records)

    job_matches = load_persistent_match_cache(
        db,
        student_record,
        job_version,
        "llm",
    )
    match_source = "llm"
    match_cached = job_matches is not None

    if job_matches is None:
        job_matches = load_persistent_match_cache(
            db,
            student_record,
            job_version,
            "local",
        )
        match_source = "local"
        match_cached = job_matches is not None

    if job_matches is None:
        job_matches = calculate_job_match(
            student_data,
            job_records,
            assessment=assessment,
            top_n=top_n,
        )
        save_persistent_match_cache(
            db,
            student_record,
            job_version,
            "local",
            job_matches,
        )
        match_source = "local"
        match_cached = False

    return attach_cached_gap_paths(student_record, job_matches), match_source, match_cached


def normalize_search_text(value: str | None) -> str:
    return re.sub(r"[\s\-_（）()【】\[\]·.,，、/|:：;；]+", "", str(value or "").lower())


def extract_company_job_terms(job_name: str) -> list[str]:
    text = str(job_name or "").strip()
    terms: list[str] = []
    for item in re.findall(r"[A-Za-z0-9+#.]+", text):
        if len(item.strip()) >= 2:
            terms.append(item.strip())

    keyword_terms = [
        "软件开发",
        "后端",
        "前端",
        "云计算",
        "数据库",
        "数据分析",
        "数据",
        "算法",
        "测试",
        "运维",
        "开发",
        "工程师",
        "实习",
        "Java",
        "Python",
        "C++",
        "AI",
        "DevOps",
        "SQL",
        "MySQL",
    ]
    normalized = normalize_search_text(text)
    for term in keyword_terms:
        if normalize_search_text(term) in normalized:
            terms.append(term)

    return list(dict.fromkeys(term for term in terms if term))


def company_job_title_score(target_job_name: str, candidate_job_name: str) -> int:
    target_norm = normalize_search_text(target_job_name)
    candidate_norm = normalize_search_text(candidate_job_name)
    if not target_norm or not candidate_norm:
        return 0
    if target_norm == candidate_norm:
        return 100
    if target_norm in candidate_norm or candidate_norm in target_norm:
        return 88

    terms = extract_company_job_terms(target_job_name)
    if not terms:
        return 0

    score = 0
    for term in terms:
        term_norm = normalize_search_text(term)
        if not term_norm:
            continue
        if term_norm in candidate_norm:
            if term_norm in {"工程师", "开发", "实习"}:
                score += 8
            else:
                score += 18
    return min(score, 82)


def build_company_candidate_payload(record: JobKnowledgeRecord) -> dict:
    return {
        "id": record.id,
        "job_id": record.id,
        "job_name": record.job_name,
        "company_name": record.company_name,
        "hiring_city": record.hiring_city,
        "educational_requirements": record.educational_requirements,
        "required_skills_json": record.required_skills_json,
        "related_projects_json": record.related_projects_json,
        "recommended_courses_json": record.recommended_courses_json,
        "recommended_certificates_json": record.recommended_certificates_json,
        "salary_range": record.salary_range,
    }


def get_company_candidate_records(
    db: Session,
    target_job_name: str,
    max_candidates: int = 120,
) -> list[JobKnowledgeRecord]:
    base_query = (
        db.query(JobKnowledgeRecord)
        .filter(JobKnowledgeRecord.company_name != "")
    )
    target_job_name = str(target_job_name or "").strip()
    records: list[JobKnowledgeRecord] = []
    seen_ids: set[int] = set()

    if target_job_name:
        direct_matches = (
            base_query
            .filter(JobKnowledgeRecord.job_name.like(f"%{target_job_name}%"))
            .order_by(JobKnowledgeRecord.id.asc())
            .limit(max_candidates)
            .all()
        )
        for record in direct_matches:
            records.append(record)
            seen_ids.add(record.id)

        if len(records) >= 5:
            return records[:max_candidates]

        term_filters = []
        for term in extract_company_job_terms(target_job_name):
            if term == "工程师":
                continue
            term_filters.append(JobKnowledgeRecord.job_name.like(f"%{term}%"))
        if term_filters and len(records) < max_candidates:
            term_matches = (
                base_query
                .filter(or_(*term_filters))
                .order_by(JobKnowledgeRecord.id.asc())
                .limit(max_candidates)
                .all()
            )
            for record in term_matches:
                if record.id not in seen_ids:
                    records.append(record)
                    seen_ids.add(record.id)
                if len(records) >= max_candidates:
                    break

    if not records:
        records = (
            base_query
            .order_by(JobKnowledgeRecord.id.asc())
            .limit(max_candidates)
            .all()
        )

    return records[:max_candidates]


def build_company_growth_score(base_match: dict, title_score: int) -> int:
    matched_count = len(base_match.get("matched_skills") or [])
    missing_count = len(base_match.get("missing_skills") or base_match.get("skill_gaps") or [])
    skill_score = min(45, matched_count * 8)
    gap_score = max(0, 35 - missing_count * 5)
    title_component = round(title_score * 0.2)
    return max(35, min(100, skill_score + gap_score + title_component))


def build_top_company_matches_for_job(
    db: Session,
    student_record: DiagnosisRecord,
    target_job_name: str,
    top_n: int = 5,
) -> list[dict]:
    student_data = build_student_data(student_record)
    assessment = build_match_assessment(student_record)
    candidate_records = get_company_candidate_records(db, target_job_name)
    if not candidate_records:
        return []

    best_by_company: dict[str, dict] = {}
    for record in candidate_records:
        payload = build_company_candidate_payload(record)
        base_match = match_profile_to_job_local(
            student_data,
            payload,
            assessment=assessment,
        )
        title_score = company_job_title_score(target_job_name, record.job_name)
        if target_job_name and title_score <= 0 and len(candidate_records) > top_n:
            continue

        student_to_job_score = max(
            0,
            min(
                100,
                round(float(base_match.get("match_score") or 0) * 0.72 + title_score * 0.28),
            ),
        )
        job_to_student_score = build_company_growth_score(base_match, title_score)
        final_score = max(
            0,
            min(
                100,
                round(student_to_job_score * 0.6 + job_to_student_score * 0.4),
            ),
        )
        company_name = record.company_name.strip()
        company_item = {
            "job_record_id": record.id,
            "company_name": company_name,
            "job_name": record.job_name,
            "hiring_city": record.hiring_city,
            "salary_range": record.salary_range,
            "education_requirement": record.educational_requirements,
            "match_score": final_score,
            "student_to_job_score": student_to_job_score,
            "job_to_student_score": job_to_student_score,
            "title_match_score": title_score,
            "matched_skills": base_match.get("matched_skills") or [],
            "missing_skills": (base_match.get("missing_skills") or base_match.get("skill_gaps") or [])[:8],
            "recommend_reason": (
                f"系统按岗位名相关度、学生适岗分和岗位适生分进行双向筛选。"
                f"学生适岗分{student_to_job_score}，岗位适生分{job_to_student_score}。"
            ),
        }
        existed = best_by_company.get(company_name)
        if existed is None or company_item["match_score"] > existed["match_score"]:
            best_by_company[company_name] = company_item

    companies = sorted(
        best_by_company.values(),
        key=lambda item: (
            item["match_score"],
            item["student_to_job_score"],
            item["job_to_student_score"],
        ),
        reverse=True,
    )
    return companies[:top_n]


def ensure_gap_paths_for_job_matches(
    db: Session,
    student_record: DiagnosisRecord,
    job_matches: list[dict],
) -> list[dict]:
    """
    PDF 导出前补齐 TOP5 的岗位差距和路径规划。
    这里默认走本地路径规划，避免导出按钮因为 LLM 网络波动卡住。
    """
    top_jobs = job_matches[:5]
    if not top_jobs:
        return []

    agent_result = load_agent_result(student_record)
    cached_paths = agent_result.get("top5_gap_paths", [])
    if isinstance(cached_paths, dict):
        cached_paths = cached_paths.get("top5_gap_paths", [])
    if not isinstance(cached_paths, list):
        cached_paths = []

    cached_by_name = {
        item.get("job_name"): item
        for item in cached_paths
        if isinstance(item, dict) and item.get("job_name")
    }
    missing_jobs = [
        job for job in top_jobs
        if job.get("job_name") and job.get("job_name") not in cached_by_name
    ]

    if missing_jobs:
        path_result = generate_top5_gap_paths(
            student_data=build_student_data(student_record),
            job_recommendations=missing_jobs,
            use_llm=False,
        )
        generated_paths = path_result.get("top5_gap_paths", [])
        for path in generated_paths:
            if isinstance(path, dict) and path.get("job_name"):
                path["used_llm"] = bool(path_result.get("used_llm"))
                cached_by_name[path["job_name"]] = path

        kept_other_paths = [
            item for item in cached_paths
            if isinstance(item, dict) and item.get("job_name") not in cached_by_name
        ]
        agent_result["top5_gap_paths"] = kept_other_paths + list(cached_by_name.values())
        if path_result.get("agent_warning"):
            agent_result["path_agent_warning"] = path_result.get("agent_warning", "")
        student_record.agent_result_json = json.dumps(agent_result, ensure_ascii=False)
        db.add(student_record)
        db.commit()

    return attach_cached_gap_paths(student_record, top_jobs)


def build_ability_profile_export_payload(
    db: Session,
    record: DiagnosisRecord,
) -> dict:
    student_data = build_student_data(record)
    ability_scores = build_ability_scores(record)
    raw_agent_result = load_agent_result(record)
    agent_result = normalize_ability_profile_result(
        student_data=student_data,
        ability_scores=ability_scores,
        agent_result=raw_agent_result,
    )
    agent_result = sanitize_ability_profile_display(agent_result)

    top5_matches, match_source, match_cached = get_job_matches_for_record(
        db,
        record,
        top_n=10,
    )
    radar_data = [
        {"key": "professional", "name": "专业基础能力", "score": ability_scores.get("professional", 0)},
        {"key": "practice", "name": "技术实践能力", "score": ability_scores.get("practice", 0)},
        {"key": "tools", "name": "工具技能能力", "score": ability_scores.get("tools", 0)},
        {"key": "career", "name": "职业发展能力", "score": ability_scores.get("career", 0)},
    ]

    return {
        "export_type": "ability_profile",
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_id": record.id,
        "student": student_data,
        "ability_scores": ability_scores,
        "radar_data": radar_data,
        "profile": {
            "summary": agent_result.get("summary", ""),
            "profile_tags": agent_result.get("profile_tags", []),
            "advantages": agent_result.get("advantages", []),
            "weaknesses": agent_result.get("weaknesses", []),
            "dimension_insights": agent_result.get("dimension_insights", []),
            "evidence_cards": agent_result.get("evidence_cards", []),
            "development_focus": agent_result.get("development_focus", []),
            "risk_flags": agent_result.get("risk_flags", []),
            "quality_review": agent_result.get("quality_review", []),
        },
        "top5_job_matches": top5_matches[:5],
        "top5_match_meta": {
            "source": match_source,
            "cached": match_cached,
            "algorithm_version": MATCH_CACHE_ALGORITHM_VERSION,
        },
        "agent_collaboration": {
            "agent_roster": agent_result.get("agent_roster", []),
            "workflow_steps": agent_result.get("workflow_steps", []),
            "tool_calls": agent_result.get("tool_calls", []),
            "collaboration_log": agent_result.get("collaboration_log", []),
            "review_findings": agent_result.get("review_findings", []),
        },
    }


def write_ability_profile_json_to_desktop(payload: dict, student_name: str | None) -> Path:
    desktop = get_desktop_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = safe_filename_part(student_name, "student")
    file_path = desktop / f"ability_profile_{name_part}_{timestamp}.json"
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return file_path


def find_browser_pdf_executable() -> str | None:
    candidates = [
        os.getenv("PDF_BROWSER_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def render_html_to_pdf(html: str, output_path: Path) -> None:
    browser = find_browser_pdf_executable()
    if not browser:
        raise RuntimeError("未找到 Chrome 或 Edge，无法生成 PDF。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job_match_pdf_") as temp_dir:
        html_path = Path(temp_dir) / "top5_job_match_report.html"
        html_path.write_text(html, encoding="utf-8")
        command = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            f"--print-to-pdf={str(output_path)}",
            html_path.resolve().as_uri(),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60,
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"PDF 生成失败：{detail or '浏览器未成功输出文件'}")


def _html(value) -> str:
    return html_escape(str(value or ""), quote=True)


def _render_pdf_list(items, empty_text: str = "暂无") -> str:
    if items is None:
        values = []
    elif isinstance(items, (list, tuple, set)):
        values = list(items)
    else:
        values = [items]

    cleaned = []
    for item in values:
        if isinstance(item, dict):
            text = "；".join(
                f"{key}：{value}" for key, value in item.items()
                if value not in (None, "", [])
            )
        else:
            text = str(item or "").strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        return f"<p class=\"muted\">{_html(empty_text)}</p>"
    return "<ul>" + "".join(f"<li>{_html(item)}</li>" for item in cleaned) + "</ul>"


def _pdf_text_items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return _as_text_list(value)


def build_job_match_pdf_html(
    student_data: dict,
    job_matches: list[dict],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    student_rows = [
        ("姓名", student_data.get("name")),
        ("专业", student_data.get("major")),
        ("年级", student_data.get("grade")),
        ("目标岗位", student_data.get("target_job")),
        ("掌握技能", student_data.get("skills")),
    ]
    summary_rows = []
    path_sections = []

    for index, job in enumerate(job_matches[:5], start=1):
        gap_skills = _pdf_text_items(
            job.get("missing_skills") if "missing_skills" in job else job.get("skill_gaps")
        )
        matched_skills = _pdf_text_items(job.get("matched_skills"))
        detail = job.get("gap_detail") or {}
        summary_rows.append(f"""
            <tr>
                <td>{index}</td>
                <td>{_html(job.get("job_name"))}</td>
                <td>{_html(job.get("match_score"))}%</td>
                <td>{_html("、".join(matched_skills) or "暂无明显匹配技能")}</td>
                <td>{_html("、".join(gap_skills or []) or "暂无明显短板")}</td>
                <td>{_html(job.get("recommend_reason") or job.get("description") or "当前岗位与学生已有能力存在一定匹配度。")}</td>
            </tr>
        """)

        stage_cards = []
        for stage in detail.get("learning_stages") or []:
            actions = stage.get("actions") or stage.get("tasks") or []
            deliverables = stage.get("deliverables")
            if not deliverables and stage.get("deliverable"):
                deliverables = [stage.get("deliverable")]
            stage_cards.append(f"""
                <div class="stage-card">
                    <h4>{_html(stage.get("stage") or "学习阶段")}</h4>
                    <p class="duration">{_html(stage.get("duration"))}</p>
                    <p><strong>目标：</strong>{_html(stage.get("goal"))}</p>
                    <div><strong>行动任务：</strong>{_render_pdf_list(actions)}</div>
                    <div><strong>阶段成果：</strong>{_render_pdf_list(deliverables)}</div>
                </div>
            """)

        path_sections.append(f"""
            <section class="job-section">
                <h3>{index}. {_html(job.get("job_name"))}：岗位差距与路径规划</h3>
                <p class="path-summary">{_html(detail.get("path_summary") or "已根据当前岗位匹配结果生成补齐建议。")}</p>

                <div class="block">
                    <h4>当前差距清单</h4>
                    {_render_pdf_list(detail.get("gap_list"))}
                </div>

                <div class="block">
                    <h4>推荐项目</h4>
                    {_render_pdf_list(detail.get("recommended_projects"))}
                </div>

                <div class="block">
                    <h4>学习阶段</h4>
                    <div class="stage-grid">
                        {''.join(stage_cards) if stage_cards else '<p class="muted">暂无学习阶段规划。</p>'}
                    </div>
                </div>
            </section>
        """)

    student_table_rows = "".join(
        f"<tr><th>{_html(label)}</th><td>{_html(value)}</td></tr>"
        for label, value in student_rows
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>TOP5 岗位匹配与路径规划报告</title>
    <style>
        @page {{ size: A4; margin: 18mm 16mm; }}
        body {{
            margin: 0;
            color: #111827;
            font-family: "Microsoft YaHei", "SimSun", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.7;
        }}
        h1 {{ margin: 0 0 8px; font-size: 24px; }}
        h2 {{ margin: 26px 0 12px; font-size: 18px; color: #0f3768; }}
        h3 {{ margin: 0 0 10px; font-size: 16px; color: #0f3768; }}
        h4 {{ margin: 0 0 8px; font-size: 14px; color: #111827; }}
        table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
        th, td {{ border: 1px solid #d8e0eb; padding: 8px 10px; vertical-align: top; word-break: break-word; }}
        th {{ background: #f1f5f9; text-align: left; }}
        ul {{ margin: 4px 0 0; padding-left: 18px; }}
        li {{ margin-bottom: 4px; }}
        .meta {{ color: #64748b; margin-bottom: 18px; }}
        .student-table th {{ width: 96px; }}
        .job-section {{
            page-break-inside: avoid;
            margin-top: 18px;
            padding: 14px 16px;
            border: 1px solid #d8e0eb;
            border-radius: 8px;
            background: #fbfdff;
        }}
        .path-summary {{
            margin: 0 0 12px;
            padding: 10px 12px;
            border-radius: 6px;
            background: #eef6ff;
            color: #1e3a8a;
        }}
        .block {{ margin-top: 12px; }}
        .stage-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .stage-card {{
            padding: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            background: #ffffff;
        }}
        .duration {{ margin: 0 0 6px; color: #64748b; }}
        .muted {{ color: #64748b; margin: 0; }}
    </style>
</head>
<body>
    <h1>TOP5 岗位匹配与路径规划报告</h1>
    <p class="meta">生成时间：{_html(generated_at)}；报告内容来自系统岗位数据表相似度匹配结果及对应路径规划。</p>

    <h2>学生信息</h2>
    <table class="student-table">{student_table_rows}</table>

    <h2>TOP5 推荐岗位</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 36px;">序号</th>
                <th style="width: 110px;">岗位</th>
                <th style="width: 62px;">匹配度</th>
                <th>已匹配技能</th>
                <th>待补齐技能</th>
                <th>推荐理由</th>
            </tr>
        </thead>
        <tbody>{''.join(summary_rows)}</tbody>
    </table>

    <h2>岗位差距明细与补齐路径</h2>
    {''.join(path_sections)}
</body>
</html>"""


def write_job_match_pdf_to_desktop(
    student_data: dict,
    job_matches: list[dict],
) -> Path:
    desktop = get_desktop_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = safe_filename_part(student_data.get("name"), "student")
    file_path = desktop / f"top5_job_match_path_{name_part}_{timestamp}.pdf"
    html = build_job_match_pdf_html(student_data, job_matches)
    render_html_to_pdf(html, file_path)
    return file_path


def normalize_ability_profile_result(
    student_data: dict,
    ability_scores: dict,
    agent_result: dict
) -> dict:
    if not agent_result:
        return {}

    has_new_profile = bool(
        agent_result.get("dimension_insights")
        or agent_result.get("workflow_steps")
    )

    if has_new_profile:
        normalized = dict(agent_result)
        normalized["agent_roster"] = normalized.get("agent_roster") or AGENT_ROSTER
        return normalized

    return {}


REMOVED_PROFILE_SECTION_TERMS = (
    "成长规划智能体",
    "成长路径",
    "细化任务",
    "growth_path",
    "learning_tasks",
    "growth_plan",
    "task_decomposition",
    "TaskDecompositionTool",
)


def _contains_removed_profile_section(value) -> bool:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "")
    return any(term in text for term in REMOVED_PROFILE_SECTION_TERMS)


def sanitize_ability_profile_display(agent_result: dict) -> dict:
    if not agent_result:
        return {}

    sanitized = dict(agent_result)
    sanitized.pop("growth_path", None)
    sanitized.pop("learning_tasks", None)

    for key in ("agent_roster", "llm_agents"):
        values = sanitized.get(key, [])
        if isinstance(values, list):
            sanitized[key] = [
                item for item in values
                if not _contains_removed_profile_section(item)
            ]

    for key in ("tool_calls", "collaboration_log", "workflow_steps"):
        values = sanitized.get(key, [])
        if isinstance(values, list):
            sanitized[key] = [
                item for item in values
                if not _contains_removed_profile_section(item)
            ]

    return sanitized


COMMON_RESUME_SKILLS = [
    "Java", "Spring Boot", "Spring Cloud", "MyBatis", "MySQL", "Redis", "Linux",
    "Git", "Maven", "Docker", "Nginx", "RabbitMQ", "Kafka", "Python", "FastAPI",
    "Django", "Flask", "Vue", "React", "JavaScript", "TypeScript", "HTML", "CSS",
    "SQL", "数据结构", "算法", "操作系统", "计算机网络", "数据库", "设计模式",
    "性能优化", "缓存", "微服务", "分布式", "高并发", "机器学习", "深度学习", "大模型",
]


def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            for separator in [",", "，", ";", "；", "\n", "\r", "|", "/"]:
                text = text.replace(separator, "、")
            return [item.strip() for item in text.split("、") if item.strip()]
        return _as_text_list(loaded)
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(_as_text_list(item))
        return values
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_as_text_list(item))
        return [item for index, item in enumerate(result) if item and item not in result[:index]]
    text = str(value).strip()
    return [text] if text else []


def _first_regex_group(text: str, patterns: list[str], default: str = "无") -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ：:，,；;|")
            if value:
                return value[:120]
    return default


def _infer_target_job_from_text(text: str) -> str:
    explicit = _first_regex_group(
        text,
        [
            r"(?:求职意向|目标岗位|应聘岗位|意向岗位|期望岗位)\s*[：:]\s*([^\n\r]{2,80})",
            r"(?:岗位)\s*[：:]\s*([^\n\r]{2,80})",
        ],
        ""
    )
    if explicit:
        return explicit

    lower_text = text.lower()
    if ("java" in lower_text or "spring" in lower_text) and ("后端" in text or "开发" in text):
        return "Java 后端工程师"
    if any(keyword in lower_text for keyword in ["python", "fastapi", "django", "flask"]):
        return "Python 后端工程师"
    if any(keyword in lower_text for keyword in ["vue", "react", "javascript", "typescript"]) or "前端" in text:
        return "前端开发工程师"
    if any(keyword in text for keyword in ["算法", "机器学习", "深度学习", "大模型"]):
        return "算法工程师"
    return "软件开发工程师"


def extract_student_profile_locally(message: str, resume_text: str) -> dict[str, str]:
    text = re.sub(r"\r\n?", "\n", resume_text or "")
    compact_text = re.sub(r"[ \t]+", " ", text)
    name = _first_regex_group(
        compact_text,
        [
            r"(?:姓名|Name)\s*[：:]\s*([^\n\r]{2,20})",
            r"^\s*([一-龥]{2,4})\s*$",
        ],
        "无"
    )
    major = _first_regex_group(
        compact_text,
        [
            r"(?:专业|所学专业)\s*[：:]\s*([^\n\r]{2,60})",
            r"([一-龥A-Za-z0-9]{2,40}(?:工程|科学与技术|软件|计算机|人工智能|数据科学)[一-龥A-Za-z0-9]*)",
        ],
        "无"
    )
    grade = _first_regex_group(
        compact_text,
        [
            r"(?:年级|学历|教育背景)\s*[：:]\s*([^\n\r]{2,40})",
            r"(本科|硕士|研究生|博士|大一|大二|大三|大四|应届生)",
        ],
        "本科" if "本科" in compact_text else "无"
    )
    skills = [
        skill for skill in COMMON_RESUME_SKILLS
        if re.search(re.escape(skill), compact_text, flags=re.IGNORECASE)
    ]
    project_lines = [
        line.strip()
        for line in text.splitlines()
        if any(keyword in line for keyword in ["项目", "系统", "平台", "实习", "开发", "比赛", "竞赛"])
    ]
    certificate_lines = [
        line.strip()
        for line in text.splitlines()
        if any(keyword in line for keyword in ["证书", "认证", "英语", "CET", "四级", "六级", "软考"])
    ]
    competition_lines = [
        line.strip()
        for line in text.splitlines()
        if any(keyword in line for keyword in ["竞赛", "比赛", "获奖", "蓝桥杯", "ACM"])
    ]

    return {
        "name": name[:50],
        "major": major[:100],
        "grade": grade[:30],
        "target_job": _infer_target_job_from_text(f"{message}\n{text}")[:100],
        "skills": "、".join(skills) if skills else "无",
        "projects": "\n".join(project_lines[:8])[:1200] or "无",
        "competitions": "\n".join(competition_lines[:5])[:600] or "无",
        "certificates": "\n".join(certificate_lines[:5])[:600] or "无",
        "self_intro": compact_text[:800] if compact_text else "无",
    }


def _score_level_label(score: int) -> str:
    if score >= 85:
        return "优势明显"
    if score >= 70:
        return "基础较稳"
    if score >= 55:
        return "已有基础"
    if score >= 40:
        return "需要补强"
    return "证据不足"


def _trim_list(values: list[str], limit: int, fallback: str) -> list[str]:
    cleaned = [str(item).strip() for item in values if str(item or "").strip() and str(item).strip() != "无"]
    return cleaned[:limit] if cleaned else [fallback]


def _career_stage_titles(job_name: str) -> list[str]:
    normalized = str(job_name or "").lower()
    if "java" in normalized and "后端" in job_name:
        return ["初级Java后端工程师", "Java后端工程师", "高级Java后端工程师", "后端技术专家", "系统架构师", "技术负责人"]
    if "python" in normalized and "后端" in job_name:
        return ["初级Python后端工程师", "Python后端工程师", "高级Python后端工程师", "后端技术专家", "平台架构师", "技术负责人"]
    if "前端" in job_name or "vue" in normalized or "react" in normalized:
        return ["初级前端工程师", "前端开发工程师", "高级前端工程师", "前端技术专家", "前端架构师", "技术负责人"]
    if "算法" in job_name or "机器学习" in job_name or "大模型" in job_name:
        return ["初级算法工程师", "算法工程师", "高级算法工程师", "算法专家", "算法架构师", "算法负责人"]
    base_job = job_name or "目标岗位"
    return [f"初级{base_job}", base_job, f"高级{base_job}", "核心骨干", "领域专家", "团队负责人"]


def select_top1_guidance_job(student_data: dict, assessment: dict, job_records: list | None) -> dict:
    if job_records:
        matches = []
        for record in job_records:
            match = match_profile_to_job_local(student_data, record, assessment)
            matches.append(match)
        matches.sort(key=lambda item: item.get("match_score", 0), reverse=True)
        if matches:
            top_job = matches[0]
            return {
                "job_name": top_job.get("job_name") or student_data.get("target_job") or "目标岗位",
                "match_score": top_job.get("match_score", 0),
                "matched_skills": _as_text_list(top_job.get("matched_skills"))[:8],
                "skill_gaps": _as_text_list(top_job.get("skill_gaps") or top_job.get("missing_skills"))[:8],
                "recommend_reason": top_job.get("recommend_reason") or top_job.get("reason") or "",
            }

    ability_scores = assessment.get("ability_scores", {})
    fallback_score = round(sum(ability_scores.values()) / max(len(ability_scores), 1))
    return {
        "job_name": student_data.get("target_job") or "目标岗位",
        "match_score": fallback_score,
        "matched_skills": _as_text_list(assessment.get("recognized_skills"))[:8],
        "skill_gaps": [],
        "recommend_reason": "岗位库暂无可用匹配结果，系统使用简历中的求职意向作为发展趋势锚点。",
    }


def _career_projection_values(start: int, end: int = 95) -> list[int]:
    start = max(15, min(82, int(start or 0)))
    end = max(start + 10, min(100, end))
    ratios = [0, 0.16, 0.38, 0.62, 0.82, 1]
    return [round(start + (end - start) * ratio) for ratio in ratios]


def build_development_projection(ability_scores: dict, top1_job: dict) -> dict:
    labels = ["入职", "第1年", "第3年", "第5年", "第7年", "第10年"]
    job_name = top1_job.get("job_name") or "目标岗位"
    match_score = int(top1_job.get("match_score") or 0)
    professional = int(ability_scores.get("professional", 0) or 0)
    practice = int(ability_scores.get("practice", 0) or 0)
    tools = int(ability_scores.get("tools", 0) or 0)
    career = int(ability_scores.get("career", 0) or 0)
    current_average = round((professional + practice + tools + career) / 4)
    position_titles = _career_stage_titles(job_name)
    stage_years = ["0-1年", "1年", "3年", "5年", "7年", "10年"]
    stage_notes = [
        "完成岗位入门，能在指导下交付清晰模块。",
        "独立负责常规需求，形成稳定编码、联调和问题排查能力。",
        "负责核心模块，能处理性能、稳定性和复杂业务问题。",
        "主导子系统或关键项目，开始承担技术方案设计。",
        "沉淀平台能力和工程规范，影响多个项目或小团队。",
        "具备路线规划、架构决策和团队带教能力。",
    ]

    return {
        "labels": labels,
        "anchor_job": {
            "job_name": job_name,
            "match_score": match_score,
            "matched_skills": top1_job.get("matched_skills", []),
            "skill_gaps": top1_job.get("skill_gaps", []),
            "recommend_reason": top1_job.get("recommend_reason", ""),
        },
        "position_path": [
            {
                "label": labels[index],
                "year": stage_years[index],
                "title": position_titles[index],
                "note": stage_notes[index],
                "level_score": [20, 34, 50, 66, 82, 94][index],
            }
            for index in range(len(labels))
        ],
        "series": [
            {"key": "career_level", "name": "职位层级", "values": [20, 34, 50, 66, 82, 94], "color": "#2563eb"},
            {"key": "job_competence", "name": "岗位胜任力", "values": _career_projection_values(max(match_score, current_average), 96), "color": "#0f766e"},
            {"key": "technical_depth", "name": "技术深度", "values": _career_projection_values(round((professional + tools) / 2), 94), "color": "#f97316"},
            {"key": "project_influence", "name": "项目影响力", "values": _career_projection_values(practice, 92), "color": "#7c3aed"},
            {"key": "career_readiness", "name": "职业成熟度", "values": _career_projection_values(career, 95), "color": "#dc2626"},
        ],
    }


def build_employment_guidance_from_resume_text(
    message: str,
    resume_text: str,
    job_records: list | None = None
) -> dict:
    student_data = extract_student_profile_locally(message, resume_text)
    student_data["resume_text"] = resume_text
    assessment = score_four_dimensions_local(student_data)
    ability_scores = assessment["ability_scores"]
    score_evidence = assessment.get("score_evidence", {})
    top1_job = select_top1_guidance_job(student_data, assessment, job_records)
    student_data["target_job"] = top1_job.get("job_name") or student_data.get("target_job")
    career_projection = build_development_projection(ability_scores, top1_job)

    recognized_skills = _as_text_list(assessment.get("recognized_skills")) or _as_text_list(student_data.get("skills"))
    top1_matched_skills = _as_text_list(top1_job.get("matched_skills")) or recognized_skills
    top1_skill_gaps = _as_text_list(top1_job.get("skill_gaps"))
    project_evidence = _trim_list(
        [line.strip() for line in str(student_data.get("projects", "")).splitlines()],
        5,
        "简历中项目经历证据不足，需要补充项目名称、职责、技术栈和成果。"
    )
    certificate_evidence = _trim_list(
        [line.strip() for line in str(student_data.get("certificates", "")).splitlines()],
        4,
        "简历中证书或认证信息较少，可补充语言等级、专业认证或课程证书。"
    )
    competition_evidence = _trim_list(
        [line.strip() for line in str(student_data.get("competitions", "")).splitlines()],
        4,
        "简历中竞赛/获奖信息较少，如有课程设计、比赛或训练营成果建议补充。"
    )
    dimensions = [
        {"key": "professional", "name": "专业基础", "score": ability_scores.get("professional", 0)},
        {"key": "practice", "name": "技术实践", "score": ability_scores.get("practice", 0)},
        {"key": "tools", "name": "工具技能", "score": ability_scores.get("tools", 0)},
        {"key": "career", "name": "职业发展", "score": ability_scores.get("career", 0)},
    ]
    weakest = sorted(dimensions, key=lambda item: item["score"])[:2]
    strongest = sorted(dimensions, key=lambda item: item["score"], reverse=True)[:2]
    target_job = student_data.get("target_job") or "目标岗位"
    ten_year_target = career_projection.get("position_path", [{}])[-1].get("title", "技术负责人")
    skills_text = "、".join(recognized_skills[:8]) if recognized_skills else "简历中可识别技能较少"
    top1_matched_text = "、".join(top1_matched_skills[:8]) if top1_matched_skills else skills_text
    top1_gap_text = "、".join(top1_skill_gaps[:6]) if top1_skill_gaps else "暂无明显岗位技能缺口，重点提升项目复杂度和业务影响力"
    weakest_text = "、".join(item["name"] for item in weakest)
    strongest_text = "、".join(item["name"] for item in strongest)

    evidence_basis = [
        {
            "title": "TOP1岗位锚点依据",
            "conclusion": f"长期发展趋势以推荐TOP1“{target_job}”为锚点，当前匹配度为 {top1_job.get('match_score', 0)}%。",
            "items": [
                f"TOP1岗位：{target_job}",
                f"已匹配岗位技能：{top1_matched_text}",
                f"后续关键补强：{top1_gap_text}",
                f"四维能力中相对优势为：{strongest_text}；优先补强：{weakest_text}。",
            ]
        },
        {
            "title": "项目证据依据",
            "conclusion": "就业指导优先看可验证项目证据：是否有业务场景、个人职责、技术难点、量化结果和可展示产物。",
            "items": project_evidence,
        },
        {
            "title": "能力评分依据",
            "conclusion": "四维分数来自简历中的技能、项目、证书、目标岗位和表达完整度。",
            "items": [
                f"专业基础：{ability_scores.get('professional', 0)} 分，{_score_level_label(ability_scores.get('professional', 0))}，证据：{'、'.join(_as_text_list(score_evidence.get('professional'))[:6]) or '课程/基础知识证据不足'}。",
                f"技术实践：{ability_scores.get('practice', 0)} 分，{_score_level_label(ability_scores.get('practice', 0))}，证据：{'、'.join(_as_text_list(score_evidence.get('practice'))[:6]) or '项目深度和成果证据不足'}。",
                f"工具技能：{ability_scores.get('tools', 0)} 分，{_score_level_label(ability_scores.get('tools', 0))}，证据：{'、'.join(_as_text_list(score_evidence.get('tools'))[:6]) or '工程工具链证据不足'}。",
                f"职业发展：{ability_scores.get('career', 0)} 分，{_score_level_label(ability_scores.get('career', 0))}，证据：{'、'.join(_as_text_list(score_evidence.get('career'))[:6]) or '目标表达和面试准备证据不足'}。",
            ]
        },
        {
            "title": "补充材料依据",
            "conclusion": "证书、竞赛和训练经历可以增强可信度，但必须和就业方向有关。",
            "items": certificate_evidence + competition_evidence,
        },
    ]

    precision_guidance = [
        {
            "title": "就业主线定位",
            "basis": [f"TOP1岗位锚点：{target_job}", f"当前可用岗位技能证据：{top1_matched_text}", f"优势能力：{strongest_text}"],
            "advice": f"把求职主线收束为“{target_job}方向的可交付型候选人”，先达成入职胜任，再逐步向“{ten_year_target}”发展。",
            "actions": ["简历开头写清目标方向、核心技能和最强项目。", "每个项目补齐业务目标、个人负责模块、技术难点和最终结果。", "删除和目标方向弱相关的泛泛表述，把空间留给岗位关键词和可验证成果。"]
        },
        {
            "title": "简历证据强化",
            "basis": project_evidence,
            "advice": "当前就业竞争力的关键不在继续堆技能，而在把已有项目写成能证明岗位能力的证据链。",
            "actions": ["为核心项目补充 2-3 个量化指标。", "把“参与/负责”改成具体动作。", "补一段项目复盘：问题、排查、取舍和改进方案。"]
        },
        {
            "title": "短板补齐顺序",
            "basis": [f"{weakest[0]['name']}当前为 {weakest[0]['score']} 分。", f"{weakest[1]['name']}当前为 {weakest[1]['score']} 分。"],
            "advice": f"优先补齐{weakest_text}，先补能直接写进简历、能在面试中讲清楚的内容。",
            "actions": ["每天固定一个短板主题，学习后产出笔记或代码提交。", "每周选择一个岗位 JD，对照检查简历证据。", "把补齐结果沉淀到项目 README、博客或作品集。"]
        },
        {
            "title": "面试准备策略",
            "basis": ["职业发展能力由目标清晰度、表达完整度、简历材料和面试准备共同决定。", f"当前职业发展分：{ability_scores.get('career', 0)} 分。"],
            "advice": "面试准备要围绕项目证据展开，避免只背技术点。",
            "actions": ["准备 1 分钟自我介绍。", "准备 3 个 STAR 项目故事。", "准备常见追问：为什么这样设计、还有什么优化空间、数据量扩大如何处理。"]
        },
    ]

    development_suggestions = [
        {
            "title": "0-1年：岗位入门与稳定交付",
            "basis": f"TOP1岗位为{target_job}，当前匹配度 {top1_job.get('match_score', 0)}%。",
            "advice": "第一年重点是做到需求理解清楚、代码质量稳定、问题能闭环。",
            "milestones": ["入职前补齐核心技能缺口：" + top1_gap_text + "。", "完成一版围绕TOP1岗位的简历和作品集。", "形成代码规范、接口文档、问题复盘三类交付习惯。"]
        },
        {
            "title": "1-3年：独立负责核心模块",
            "basis": "职位提升来自可独立承担模块，而不是只完成零散任务。",
            "advice": f"围绕{target_job}的核心业务场景，承担一个可持续迭代的模块或子系统。",
            "milestones": ["独立完成需求评审、技术方案、开发联调和上线复盘。", "积累 2-3 个能讲清楚难点、取舍和结果的项目案例。", "开始关注性能、稳定性、可维护性和团队协作成本。"]
        },
        {
            "title": "3-5年：高级岗位与技术深度",
            "basis": f"当前短板集中在{weakest_text}，中期晋升需要把短板转成可证明的技术深度。",
            "advice": "从“会做功能”提升到“能设计方案、控制风险、提升系统质量”。",
            "milestones": ["主导一次复杂需求或系统优化。", "建立技术专题能力，如性能优化、架构设计或稳定性治理。", "开始承担新人带教、代码评审和方案评审。"]
        },
        {
            "title": "5-10年：专家/负责人路径",
            "basis": f"十年趋势目标为{ten_year_target}，需要同时提升技术判断、业务理解和团队影响力。",
            "advice": "长期晋升不只看技术点数量，更看能否定义问题、组织资源并交付结果。",
            "milestones": ["沉淀负责领域的方法论和技术规范。", "推动跨模块或跨团队项目，证明影响力扩大。", "形成架构决策、团队培养和业务目标对齐能力。"]
        },
    ]

    return {
        "student": student_data,
        "ability_scores": ability_scores,
        "summary": f"本次只执行就业指导分支。系统以推荐TOP1岗位“{target_job}”作为长期发展锚点；当前匹配度为 {top1_job.get('match_score', 0)}%，十年内建议从岗位入门逐步提升到“{ten_year_target}”。",
        "top1_job": top1_job,
        "evidence_basis": evidence_basis,
        "precision_guidance": precision_guidance,
        "development_suggestions": development_suggestions,
        "trend": {
            "latest_score": round(sum(ability_scores.values()) / 4, 2),
            "weakest_dimension": weakest[0],
            "best_dimension": strongest[0],
            "anchor_job": target_job,
            "ten_year_target": ten_year_target,
        },
        "trend_chart": career_projection,
    }


def detect_agent_intent(message: str) -> str:
    text = (message or "").strip()
    if any(keyword in text for keyword in ["就业", "就业指导", "精准就业", "指导一下我的就业", "指导我的就业", "求职", "求职指导", "职业发展", "发展建议", "智能发展"]):
        return "employment"
    if any(keyword in text for keyword in ["优化", "改简历", "润色", "简历优化"]):
        return "resume"
    if any(keyword in text for keyword in ["画像", "能力", "诊断", "分析我"]):
        return "profile"
    return ""


def render_ability_profile(
    request: Request,
    record: DiagnosisRecord | None,
    *,
    generation_mode: bool = False,
):
    """
    统一渲染能力画像页面。
    """

    ability_explain = {
        "professional": "专业基础能力主要来自课程、专业知识和相关证书。",
        "practice": "技术实践能力主要来自项目经历、竞赛经历和实习经历。",
        "tools": "工具技能能力主要来自 Python、Java、Linux、数据库、AI 工具等掌握情况。",
        "career": "职业发展能力主要来自表达能力、简历质量、目标清晰度和面试准备情况。"
    }

    if record is None:
        student_data = {
            "name": "无",
            "major": "无",
            "grade": "无",
            "target_job": "无",
            "skills": "无",
            "projects": "无",
            "competitions": "无",
            "certificates": "无",
            "self_intro": "无"
        }

        ability_scores = {
            "professional": 0,
            "practice": 0,
            "tools": 0,
            "career": 0
        }
    else:
        student_data = build_student_data(record)
        ability_scores = build_ability_scores(record)

    raw_agent_result = load_agent_result(record)

    agent_result = normalize_ability_profile_result(
        student_data=student_data,
        ability_scores=ability_scores,
        agent_result=raw_agent_result
    )
    agent_result = sanitize_ability_profile_display(agent_result)

    job_recommendations = agent_result.get("job_recommendations", [])
    top5_gap_paths = agent_result.get("top5_gap_paths", [])
    if isinstance(top5_gap_paths, dict):
        top5_gap_paths = top5_gap_paths.get("top5_gap_paths", [])
    if not isinstance(top5_gap_paths, list):
        top5_gap_paths = []

    gap_map = {
        item.get("job_name"): item
        for item in top5_gap_paths
        if isinstance(item, dict)
    }

    for index, job in enumerate(job_recommendations):
        if index < 5:
            job["gap_detail"] = gap_map.get(job.get("job_name")) or {}

    return templates.TemplateResponse(
        request,
        "ability_profile.html",
        {
            "title": "学生能力画像",
            "student": student_data,
            "ability_scores": ability_scores,
            "ability_explain": ability_explain,
            "record": record,
            "generation_mode": generation_mode,
            "username": request.session.get("username"),

            "agent_result": agent_result,
            "summary": agent_result.get("summary", ""),
            "advantages": agent_result.get("advantages", []),
            "weaknesses": agent_result.get("weaknesses", []),
            "profile_tags": agent_result.get("profile_tags", []),
            "risk_flags": agent_result.get("risk_flags", []),
            "evidence_cards": agent_result.get("evidence_cards", []),
            "dimension_insights": agent_result.get("dimension_insights", []),
            "development_focus": agent_result.get("development_focus", []),
            "quality_review": agent_result.get("quality_review", []),
            "workflow_steps": agent_result.get("workflow_steps", []),
            "tool_calls": agent_result.get("tool_calls", []),
            "collaboration_log": agent_result.get("collaboration_log", []),
            "review_findings": agent_result.get("review_findings", []),
            "shared_workspace": agent_result.get("shared_workspace", {}),
            "agent_roster": agent_result.get("agent_roster", []),
            "llm_agents": agent_result.get("llm_agents", []),
            "used_llm": agent_result.get("used_llm", False),
            "agent_warning": agent_result.get("agent_warning", ""),
            "export_message": request.query_params.get("export_message", ""),
            "export_error": request.query_params.get("export_error", ""),
        }
    )
def build_growth_trend(records: list[DiagnosisRecord]) -> dict:
    """
    根据当前用户的全部历史诊断记录，生成成长轨迹数据。
    重点：综合成长变化与能力变化，默认比较“最近一次”和“上一次”。
    """

    trend_items = []

    for index, record in enumerate(records, start=1):
        average_score = round(
            (
                record.professional_score
                + record.practice_score
                + record.tools_score
                + record.career_score
            ) / 4,
            2
        )

        trend_items.append({
            "index": index,
            "record": record,
            "created_at": record.created_at,
            "professional_score": record.professional_score,
            "practice_score": record.practice_score,
            "tools_score": record.tools_score,
            "career_score": record.career_score,
            "average_score": average_score
        })

    latest_item = trend_items[-1]

    # 如果只有一次诊断，没有“上一次”，就拿自己和自己比，变化为 0
    if len(trend_items) >= 2:
        previous_item = trend_items[-2]
    else:
        previous_item = trend_items[-1]

    total_change = round(
        latest_item["average_score"] - previous_item["average_score"],
        2
    )

    professional_change = latest_item["professional_score"] - previous_item["professional_score"]
    practice_change = latest_item["practice_score"] - previous_item["practice_score"]
    tools_change = latest_item["tools_score"] - previous_item["tools_score"]
    career_change = latest_item["career_score"] - previous_item["career_score"]

    changes = [
        {
            "name": "专业基础能力",
            "change": professional_change,
            "previous": previous_item["professional_score"],
            "latest": latest_item["professional_score"]
        },
        {
            "name": "技术实践能力",
            "change": practice_change,
            "previous": previous_item["practice_score"],
            "latest": latest_item["practice_score"]
        },
        {
            "name": "工具技能能力",
            "change": tools_change,
            "previous": previous_item["tools_score"],
            "latest": latest_item["tools_score"]
        },
        {
            "name": "职业发展能力",
            "change": career_change,
            "previous": previous_item["career_score"],
            "latest": latest_item["career_score"]
        },
    ]

    best_improvement = max(changes, key=lambda x: x["change"])
    weakest_dimension = min(changes, key=lambda x: x["latest"])

    if weakest_dimension["name"] == "专业基础能力":
        next_suggestion = "建议继续补充专业课程知识、岗位理论基础和相关证书。"
    elif weakest_dimension["name"] == "技术实践能力":
        next_suggestion = "建议继续完成项目实战，积累可展示的项目经历。"
    elif weakest_dimension["name"] == "工具技能能力":
        next_suggestion = "建议重点提升 Python、数据库、Linux、AI 工具等工程工具能力。"
    else:
        next_suggestion = "建议优化简历表达、面试准备和职业目标描述。"

    return {
        "trend_items": trend_items,
        "previous_item": previous_item,
        "latest_item": latest_item,
        "diagnosis_count": len(records),
        "total_change": total_change,
        "changes": changes,
        "best_improvement": best_improvement,
        "weakest_dimension": weakest_dimension,
        "next_suggestion": next_suggestion
    }
# =========================================================
# 用户注册 / 登录路由
# =========================================================
@app.get("/register")
def register_page(request: Request):
    """
    注册页面
    """

    if request.session.get("user_id"):
        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request,
        "register.html",
        {
            "title": "用户注册",
            "error": "",
            "message": ""
        }
    )


@app.post("/register")
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    处理用户注册，密码仅保存 PBKDF2 哈希。
    """

    username = username.strip()
    password = password.strip()

    if not username or not password:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "title": "用户注册",
                "error": "用户名和密码不能为空",
                "message": ""
            }
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "title": "用户注册",
                "error": "密码至少需要 8 位。",
                "message": ""
            }
        )

    existing_user = db.scalar(
        select(User).where(User.username == username)
    )

    if existing_user is not None:
        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "title": "用户注册",
                "error": "用户名已存在，请换一个用户名",
                "message": ""
            }
        )

    user = User(
        username=username,
        password=hash_password(password)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return templates.TemplateResponse(
        request,
        "login.html",
        build_login_context(message="注册成功，请登录")
    )


@app.get("/login")
def login_page(request: Request):
    """
    登录页面
    """

    if request.session.get("user_id"):
        return RedirectResponse(
            url="/collaboration",
            status_code=303
        )

    return templates.TemplateResponse(
        request,
        "login.html",
        build_login_context()
    )


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    user_role: str = Form("student"),
    role_access_code: str = Form(""),
    db: Session = Depends(get_db)
):
    """
    处理用户登录。
    """

    username = username.strip()
    password = password.strip()
    selected_role = normalize_triparty_role(user_role)

    if not username or not password:
        return templates.TemplateResponse(
            request,
            "login.html",
            build_login_context(
                error="用户名和密码不能为空",
                selected_role=selected_role,
            )
        )

    user = db.scalar(
        select(User).where(User.username == username)
    )

    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            build_login_context(
                error="用户不存在",
                selected_role=selected_role,
            )
        )

    password_valid, needs_upgrade = verify_password(password, user.password)
    if not password_valid:
        return templates.TemplateResponse(
            request,
            "login.html",
            build_login_context(
                error="密码错误",
                selected_role=selected_role,
            )
        )

    role_error = institution_role_access_error(selected_role, role_access_code)
    if role_error:
        return templates.TemplateResponse(
            request,
            "login.html",
            build_login_context(
                error=role_error,
                selected_role=selected_role,
            )
        )

    if needs_upgrade:
        user.password = hash_password(password)
        db.add(user)
        db.commit()

    # 登录成功：保存登录状态
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    request.session["user_role"] = selected_role
    request.session["role_label"] = TRIPARTY_ROLE_LABELS[selected_role]

    return RedirectResponse(
        url="/collaboration",
        status_code=303
    )


@app.get("/logout")
def logout(request: Request):
    """
    退出登录
    """
    request.session.clear()

    return RedirectResponse(
        url="/login",
        status_code=303
    )


# =========================================================
# 页面路由
# =========================================================

@app.get("/")
def index(request: Request):
    """
    系统首页：登录后才能访问。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "岗位能力达成学生成长诊断与精准就业智能体系统",
            "username": request.session.get("username"),
            "role_label": get_session_role_label(request),
            "message": ""
        }
    )


@app.get("/collaboration")
def triparty_collaboration_dashboard(
    request: Request,
    message: str = "",
    error: str = "",
    db: Session = Depends(get_db)
):
    """
    学生-学校-企业三方数据协同工作台。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    enforce_collaboration_rate_limit(request, "dashboard", limit=90)

    return templates.TemplateResponse(
        request,
        "triparty_collaboration.html",
        build_collaboration_context(
            request,
            db,
            message=message,
            error=error,
        )
    )


@app.post("/collaboration/role")
def triparty_role_switch(
    request: Request,
    user_role: str = Form("student"),
    role_access_code: str = Form(""),
    csrf_token: str = Form(""),
):
    """
    演示场景下允许已登录用户快速切换三方身份。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    require_valid_csrf(request, csrf_token)
    enforce_collaboration_rate_limit(request, "role-switch", limit=8)

    if not ALLOW_TRIPARTY_ROLE_SWITCH:
        return build_collaboration_redirect(
            error="隐私保护模式已禁止页面内切换身份，请退出后以授权身份重新登录。"
        )

    selected_role = normalize_triparty_role(user_role)
    role_error = institution_role_access_error(selected_role, role_access_code)
    if role_error:
        return build_collaboration_redirect(error=role_error)
    request.session["user_role"] = selected_role
    request.session["role_label"] = TRIPARTY_ROLE_LABELS[selected_role]

    return build_collaboration_redirect(
        message=f"已切换为{TRIPARTY_ROLE_LABELS[selected_role]}身份。"
    )


@app.post("/collaboration/student/submit")
def triparty_student_submit(
    request: Request,
    student_name: str = Form(...),
    major: str = Form(""),
    target_company: str = Form(...),
    target_job: str = Form(...),
    resume_text: str = Form(...),
    privacy_consent: str = Form(""),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db)
):
    """
    学生投递简历：数据先写入数据库，状态进入学校审核。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    role_error = ensure_triparty_role(request, "student")
    if role_error:
        return role_error

    require_valid_csrf(request, csrf_token)
    enforce_collaboration_rate_limit(request, "student-submit", limit=5, window_seconds=600)

    student_name = student_name.strip()
    major = major.strip()
    target_company = target_company.strip()
    target_job = target_job.strip()
    resume_text = resume_text.strip()

    input_data = {
        "student_name": student_name,
        "major": major,
        "target_company": target_company,
        "target_job": target_job,
        "resume_text": resume_text,
    }
    if not student_name or not target_company or not target_job or not resume_text:
        return templates.TemplateResponse(
            request,
            "triparty_collaboration.html",
            build_collaboration_context(
                request,
                db,
                error="请完整填写姓名、目标企业、目标岗位和简历内容。",
                input_data=input_data,
            )
        )

    if privacy_consent != "accepted":
        return templates.TemplateResponse(
            request,
            "triparty_collaboration.html",
            build_collaboration_context(
                request,
                db,
                error="请先确认隐私授权：学校可核验原始材料，企业仅接收脱敏简历。",
                input_data=input_data,
            )
        )

    field_limits = {
        "姓名": (student_name, 100),
        "专业": (major, 120),
        "目标企业": (target_company, 160),
        "目标岗位": (target_job, 160),
        "简历内容": (resume_text, 50_000),
    }
    oversized = [label for label, (value, limit) in field_limits.items() if len(value) > limit]
    if oversized:
        return templates.TemplateResponse(
            request,
            "triparty_collaboration.html",
            build_collaboration_context(
                request,
                db,
                error=f"以下字段内容过长：{'、'.join(oversized)}。",
                input_data=input_data,
            )
        )

    record = TriPartyResumeApplicationRecord(
        student_user_id=request.session.get("user_id"),
        student_username=request.session.get("username", ""),
        student_name=student_name,
        major=major,
        target_company=target_company,
        target_job=target_job,
        resume_text=resume_text,
        resume_hash=hashlib.sha256(resume_text.encode("utf-8")).hexdigest(),
        status="school_review",
    )

    db.add(record)
    db.commit()

    return build_collaboration_redirect(
        message="简历已写入数据库，并送达学校端等待老师审核。"
    )


@app.post("/collaboration/{application_id}/resume")
def triparty_secure_resume_view(
    application_id: int,
    request: Request,
    csrf_token: str = Form(""),
    db: Session = Depends(get_db),
):
    """按需返回简历；企业只能看到去除直接身份标识后的版本。"""
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return redirect_response

    require_valid_csrf(request, csrf_token)
    enforce_collaboration_rate_limit(request, "resume-view", limit=12)

    record = db.get(TriPartyResumeApplicationRecord, application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="协同记录不存在")

    role = get_session_role(request)
    user_id = request.session.get("user_id")
    if role == "student" and record.student_user_id != user_id:
        raise HTTPException(status_code=403, detail="无权查看其他学生的简历。")
    if role == "enterprise" and record.status not in {"enterprise_review", "hired", "rejected"}:
        raise HTTPException(status_code=403, detail="该材料尚未由学校转交企业。")

    redacted = role == "enterprise"
    display_name = candidate_privacy_label(record) if redacted else record.student_name
    resume_text = (
        redact_sensitive_resume_text(record.resume_text, record.student_name)
        if redacted
        else record.resume_text
    )
    logger.info(
        "collaboration resume viewed role=%s user_id=%s application_id=%s redacted=%s",
        role,
        user_id,
        application_id,
        redacted,
    )
    return templates.TemplateResponse(
        request,
        "triparty_resume_view.html",
        {
            "title": "安全查看简历",
            "role_label": get_session_role_label(request),
            "display_name": display_name,
            "target_company": record.target_company,
            "target_job": record.target_job,
            "resume_text": resume_text,
            "redacted": redacted,
        },
    )


@app.post("/collaboration/school/{application_id}/review")
def triparty_school_review(
    application_id: int,
    request: Request,
    review_result: str = Form(...),
    school_feedback: str = Form(""),
    warning_message: str = Form(""),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db)
):
    """
    学校审核简历真实性：发现造假则预警学生，未发现造假则转企业。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    role_error = ensure_triparty_role(request, "school")
    if role_error:
        return role_error

    require_valid_csrf(request, csrf_token)
    enforce_collaboration_rate_limit(request, "school-review", limit=30)

    record = db.get(TriPartyResumeApplicationRecord, application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="协同记录不存在")

    if record.status != "school_review":
        return build_collaboration_redirect(
            error="该简历已经完成学校审核，无需重复处理。"
        )

    review_result = review_result.strip()
    school_feedback = school_feedback.strip()
    warning_message = warning_message.strip()
    record.school_reviewer = request.session.get("username", "")
    record.school_feedback = school_feedback

    if review_result == "fake":
        record.status = "warning_to_student"
        record.school_review_result = "fake"
        record.warning_message = warning_message or school_feedback or "学校审核发现简历存在疑似造假内容，请学生核验并重新提交真实材料。"
        record.forwarded_at = None
        db.add(record)
        db.commit()
        return build_collaboration_redirect(
            message="已通过数据库向学生发送简历造假预警。"
        )

    record.status = "enterprise_review"
    record.school_review_result = "passed"
    record.school_feedback = school_feedback or "学校已核验简历信息，未发现明显造假，已转交企业评估。"
    record.warning_message = ""
    record.forwarded_at = datetime.now()
    db.add(record)
    db.commit()

    return build_collaboration_redirect(
        message="学校审核通过，简历已通过数据库自动发送给企业。"
    )


@app.post("/collaboration/enterprise/{application_id}/decision")
def triparty_enterprise_decision(
    application_id: int,
    request: Request,
    decision: str = Form(...),
    advice_to_student: str = Form(...),
    advice_to_school: str = Form(...),
    csrf_token: str = Form(""),
    db: Session = Depends(get_db)
):
    """
    企业给出录用或拒录结果，并向学生和学校分别反馈优化建议。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    role_error = ensure_triparty_role(request, "enterprise")
    if role_error:
        return role_error

    require_valid_csrf(request, csrf_token)
    enforce_collaboration_rate_limit(request, "enterprise-decision", limit=30)

    record = db.get(TriPartyResumeApplicationRecord, application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="协同记录不存在")

    if record.status != "enterprise_review":
        return build_collaboration_redirect(
            error="该简历已经完成企业评估，无需重复处理。"
        )

    decision = decision.strip()
    advice_to_student = advice_to_student.strip()
    advice_to_school = advice_to_school.strip()

    if decision not in TRIPARTY_DECISION_LABELS:
        return build_collaboration_redirect(error="请选择录用或拒录。")

    if not advice_to_student or not advice_to_school:
        return build_collaboration_redirect(
            error="企业提交录用或拒录时，必须同时给学生和学校填写优化建议。"
        )

    record.enterprise_reviewer = request.session.get("username", "")
    record.enterprise_decision = decision
    record.status = "hired" if decision == "hire" else "rejected"
    record.enterprise_advice_to_student = advice_to_student
    record.enterprise_advice_to_school = advice_to_school
    record.decided_at = datetime.now()
    db.add(record)
    db.commit()

    return build_collaboration_redirect(
        message=f"企业已提交{TRIPARTY_DECISION_LABELS[decision]}结论，并同步给学生和学校优化建议。"
    )


@app.post("/agent/chat")
async def agent_chat(
    request: Request,
    message: str = Form(""),
    resume_file: UploadFile | None = File(None),
    db: Session = Depends(get_db)
):
    """
    首页聊天式智能体入口：根据用户指令分支到能力画像、简历优化或就业指导。
    """
    if not request.session.get("user_id"):
        return {
            "ok": False,
            "errors": ["请先登录后使用智能体。"]
        }

    message = message.strip()
    intent = detect_agent_intent(message)
    upload_warnings = []
    uploaded_filename = ""
    resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = await resume_file.read()
        resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )

    if not resume_text and len(message) > 120:
        resume_text = message

    if not intent:
        return {
            "ok": False,
            "errors": ["请告诉智能体要生成画像、优化简历还是指导就业。"],
            "warnings": upload_warnings
        }

    if not resume_text:
        return {
            "ok": False,
            "errors": ["请先拖入简历。"],
            "warnings": upload_warnings
        }

    try:
        if intent == "profile":
            student_data = extract_student_profile_from_resume(message, resume_text)
            student_context = {
                **student_data,
                "resume_text": resume_text,
                "normalized_text": resume_text,
            }
            agent_result = run_diagnosis_agent(student_context)
            agent_result["student_profile"] = {
                key: student_data[key]
                for key in ("name", "major", "grade", "target_job")
            }
            ability_scores = agent_result["ability_scores"]

            record = DiagnosisRecord(
                user_id=request.session.get("user_id"),
                name=student_data["name"],
                major=student_data["major"],
                grade=student_data["grade"],
                target_job=student_data["target_job"],
                skills=student_data["skills"],
                projects=student_data["projects"],
                competitions=student_data["competitions"],
                certificates=student_data["certificates"],
                self_intro=student_data["self_intro"],
                professional_score=ability_scores["professional"],
                practice_score=ability_scores["practice"],
                tools_score=ability_scores["tools"],
                career_score=ability_scores["career"],
                agent_status="completed",
                agent_result_json=json.dumps(agent_result, ensure_ascii=False)
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            return {
                "ok": True,
                "intent": "profile",
                "warnings": upload_warnings,
                "uploaded_filename": uploaded_filename,
                "reply": "能力画像已生成。",
                "redirect_url": f"/ability/profile/{record.id}",
                "summary": agent_result.get("summary", "")
            }

        if intent == "employment":
            job_records = db.query(JobKnowledgeRecord).all()
            result = build_employment_guidance_from_resume_text(
                message=message,
                resume_text=resume_text,
                job_records=job_records
            )
            record = EmploymentGuidanceRecord(
                user_id=request.session.get("user_id"),
                message=message,
                uploaded_filename=uploaded_filename,
                result_json=json.dumps(result, ensure_ascii=False)
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return {
                "ok": True,
                "intent": "employment",
                "warnings": upload_warnings,
                "uploaded_filename": uploaded_filename,
                "reply": "就业指导已生成。",
                "redirect_url": f"/employment/guidance?record_id={record.id}"
            }

        result = optimize_resume(
            resume_text=resume_text,
            job_description=message,
            target_role="",
            output_language="auto",
            harvard_format=False
        )
        return {
            "ok": True,
            "intent": "resume",
            "warnings": upload_warnings,
            "uploaded_filename": uploaded_filename,
            "reply": "简历优化已完成。",
            "result": result
        }
    except LLMCallError:
        return {
            "ok": False,
            "errors": ["调用LLM失败"],
            "warnings": upload_warnings
        }


@app.get("/employment/guidance")
def employment_guidance_page(
    request: Request,
    record_id: int | None = None,
    db: Session = Depends(get_db)
):
    """
    精准就业指导与智能发展建议页面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    result = None
    errors = []
    if record_id is not None:
        record = (
            db.query(EmploymentGuidanceRecord)
            .filter(
                EmploymentGuidanceRecord.id == record_id,
                EmploymentGuidanceRecord.user_id == request.session.get("user_id")
            )
            .first()
        )
        if record is None:
            errors.append("未找到本次就业指导结果，请重新生成。")
        else:
            try:
                result = json.loads(record.result_json)
            except json.JSONDecodeError:
                errors.append("本次就业指导结果读取失败，请重新生成。")

    return templates.TemplateResponse(
        request,
        "employment_guidance.html",
        {
            "title": "精准就业指导",
            "username": request.session.get("username"),
            "result": result,
            "errors": errors,
            "warnings": [],
            "input_data": {
                "message": "指导一下我的就业",
                "resume_text": "",
                "uploaded_filename": ""
            }
        }
    )


@app.post("/employment/guidance")
def employment_guidance_submit(
    request: Request,
    message: str = Form("指导一下我的就业"),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
    db: Session = Depends(get_db)
):
    """
    接收简历文本/文件，生成精准就业指导与智能发展建议。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    upload_warnings = []
    uploaded_filename = ""
    uploaded_resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = resume_file.file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )

    final_message = message.strip() or "指导一下我的就业"
    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()
    input_data = {
        "message": final_message,
        "resume_text": final_resume_text,
        "uploaded_filename": uploaded_filename
    }

    if not final_resume_text:
        return templates.TemplateResponse(
            request,
            "employment_guidance.html",
            {
                "title": "精准就业指导",
                "username": request.session.get("username"),
                "result": None,
                "errors": ["请上传可解析的简历文件，或直接粘贴简历文本。"],
                "warnings": upload_warnings,
                "input_data": input_data
            }
        )

    try:
        job_records = db.query(JobKnowledgeRecord).all()
        result = build_employment_guidance_from_resume_text(
            message=final_message,
            resume_text=final_resume_text,
            job_records=job_records
        )
        record = EmploymentGuidanceRecord(
            user_id=request.session.get("user_id"),
            message=final_message,
            uploaded_filename=uploaded_filename,
            result_json=json.dumps(result, ensure_ascii=False)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "employment_guidance.html",
            {
                "title": "精准就业指导",
                "username": request.session.get("username"),
                "result": None,
                "errors": [f"生成就业指导失败：{exc}"],
                "warnings": upload_warnings,
                "input_data": input_data
            }
        )

    return templates.TemplateResponse(
        request,
        "employment_guidance.html",
        {
            "title": "精准就业指导",
            "username": request.session.get("username"),
            "result": result,
            "errors": [],
            "warnings": upload_warnings,
            "input_data": input_data
        }
    )


@app.get("/resume/optimize")
def resume_optimize_page(request: Request):
    """
    AI 简历优化页面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return templates.TemplateResponse(
        request,
        "resume_optimize.html",
        {
            "title": "AI 优化简历",
            "username": request.session.get("username"),
            "result": None,
            "errors": [],
            "warnings": [],
            "input_data": {
                "resume_text": "",
                "job_description": "",
                "target_role": "",
                "output_language": "auto",
                "harvard_format": False,
                "uploaded_filename": ""
            }
        }
    )


@app.post("/resume/optimize")
def resume_optimize_submit(
    request: Request,
    resume_text: str = Form(""),
    job_description: str = Form(""),
    target_role: str = Form(""),
    output_language: str = Form("auto"),
    harvard_format: str | None = Form(None),
    resume_file: UploadFile | None = File(None)
):
    """
    接收简历文本/文件和招聘信息，生成 AI 优化结果。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    upload_warnings = []
    uploaded_filename = ""
    uploaded_resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = resume_file.file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )

    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()
    final_job_description = job_description.strip()
    errors = []

    if not final_resume_text:
        errors.append("请上传可解析的简历文件，或直接粘贴简历文本。")

    if not final_job_description:
        errors.append("请粘贴招聘岗位描述。")

    input_data = {
        "resume_text": resume_text.strip() or uploaded_resume_text.strip(),
        "job_description": final_job_description,
        "target_role": target_role.strip(),
        "output_language": output_language,
        "harvard_format": harvard_format == "on",
        "uploaded_filename": uploaded_filename
    }

    if errors:
        return templates.TemplateResponse(
            request,
            "resume_optimize.html",
            {
                "title": "AI 优化简历",
                "username": request.session.get("username"),
                "result": None,
                "errors": errors,
                "warnings": upload_warnings,
                "input_data": input_data
            }
        )

    try:
        result = optimize_resume(
            resume_text=final_resume_text,
            job_description=final_job_description,
            target_role=target_role,
            output_language=output_language,
            harvard_format=harvard_format == "on"
        )
    except LLMCallError:
        return templates.TemplateResponse(
            request,
            "resume_optimize.html",
            {
                "title": "AI 优化简历",
                "username": request.session.get("username"),
                "result": None,
                "errors": ["调用LLM失败"],
                "warnings": upload_warnings,
                "input_data": input_data
            }
        )

    warnings = list(upload_warnings)
    if result.get("agent_warning"):
        warnings.append(result["agent_warning"])

    return templates.TemplateResponse(
        request,
        "resume_optimize.html",
        {
            "title": "AI 优化简历",
            "username": request.session.get("username"),
            "result": result,
            "errors": [],
            "warnings": warnings,
            "input_data": input_data
        }
    )


@app.post("/resume/optimize/stream")
async def resume_optimize_stream(
    request: Request,
    resume_text: str = Form(""),
    job_description: str = Form(""),
    target_role: str = Form(""),
    output_language: str = Form("auto"),
    harvard_format: str | None = Form(None),
    resume_file: UploadFile | None = File(None)
):
    """
    Stream resume optimization progress and return the final result as NDJSON.
    The original POST endpoint remains available as a non-JS fallback.
    """
    redirect_response = get_login_redirect(request)

    def stream_event(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    if redirect_response:
        return StreamingResponse(
            iter([
                stream_event({
                    "type": "error",
                    "errors": ["请先登录后再使用简历优化。"],
                    "redirect_url": "/login",
                })
            ]),
            media_type="application/x-ndjson",
        )

    upload_warnings = []
    uploaded_filename = ""
    uploaded_resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = await resume_file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )

    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()
    final_job_description = job_description.strip()
    errors = []

    if not final_resume_text:
        errors.append("请上传可解析的简历文件，或直接粘贴简历文本。")

    if not final_job_description:
        errors.append("请粘贴招聘岗位描述。")

    input_data = {
        "resume_text": resume_text.strip() or uploaded_resume_text.strip(),
        "job_description": final_job_description,
        "target_role": target_role.strip(),
        "output_language": output_language,
        "harvard_format": harvard_format == "on",
        "uploaded_filename": uploaded_filename
    }

    def event_generator():
        yield stream_event({
            "type": "status",
            "message": "已解析简历输入，正在校验岗位信息。",
            "warnings": upload_warnings,
        })

        if errors:
            yield stream_event({
                "type": "error",
                "errors": errors,
                "warnings": upload_warnings,
                "input_data": input_data,
            })
            return

        yield stream_event({
            "type": "status",
            "message": "正在检索专家手工标注建议，并检查缓存。",
        })

        try:
            result = optimize_resume(
                resume_text=final_resume_text,
                job_description=final_job_description,
                target_role=target_role,
                output_language=output_language,
                harvard_format=harvard_format == "on",
            )
        except LLMCallError:
            yield stream_event({
                "type": "error",
                "errors": ["调用LLM失败"],
                "warnings": upload_warnings,
                "input_data": input_data,
            })
            return

        warnings = list(upload_warnings)
        if result.get("agent_warning"):
            warnings.append(result["agent_warning"])

        yield stream_event({
            "type": "status",
            "message": "命中缓存，已快速返回历史优化结果。" if result.get("cache_hit") else "优化完成，正在渲染结果。",
            "cache_hit": bool(result.get("cache_hit")),
        })
        yield stream_event({
            "type": "result",
            "result": result,
            "warnings": warnings,
            "input_data": input_data,
            "cache_hit": bool(result.get("cache_hit")),
        })

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/resume/match")
def resume_course_job_match_page(request: Request):
    """
    简历课程-岗位智能映射页面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return templates.TemplateResponse(
        request,
        "resume_match.html",
        {
            "title": "课程-岗位智能映射",
            "username": request.session.get("username"),
            "result": None,
            "errors": [],
            "warnings": [],
            "input_data": {
                "resume_text": "",
                "uploaded_filename": "",
                "top_jobs_per_course": 5,
            }
        }
    )


@app.post("/resume/match")
def resume_course_job_match_submit(
    request: Request,
    resume_text: str = Form(""),
    top_jobs_per_course: int = Form(5),
    resume_file: UploadFile | None = File(None),
    db: Session = Depends(get_db)
):
    """
    上传简历后，抽取已学课程，并生成课程-岗位映射图谱。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    upload_warnings = []
    uploaded_filename = ""
    uploaded_resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = resume_file.file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )
        upload_warnings = [
            warning.replace("，请复制简历文本到输入框。", "。").replace("；Word 可先复制文本到输入框。", "。")
            for warning in upload_warnings
        ]

    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()
    top_jobs_per_course = max(1, min(int(top_jobs_per_course or 5), 8))
    errors = []

    if not final_resume_text:
        errors.append("请上传可解析的简历文件，PDF 建议使用可复制文本版。")

    job_records = db.query(JobKnowledgeRecord).all()
    if not job_records:
        errors.append("数据库中暂无岗位数据，请先导入 data/IT岗位数据.xlsx。")

    input_data = {
        "resume_text": final_resume_text,
        "uploaded_filename": uploaded_filename,
        "top_jobs_per_course": top_jobs_per_course,
    }

    warnings = list(upload_warnings)
    result = None
    if not errors:
        inferred_ability_map, inference_warnings = build_course_inferred_ability_map(
            resume_text=final_resume_text,
            db=db,
            user_id=request.session.get("user_id"),
        )
        warnings.extend(inference_warnings)
        result = build_course_job_mapping_graph(
            resume_text=final_resume_text,
            job_records=job_records,
            top_jobs_per_course=top_jobs_per_course,
            inferred_ability_map=inferred_ability_map,
        )
        if not result["courses"]:
            errors.append("未从简历中识别到明确课程，请确认简历中包含“主要课程/相关课程”等内容，或在文本框补充课程。")
        elif not result["edges"]:
            errors.append("已识别到课程，但暂未匹配到岗位。建议先确认岗位库中的 required_skills 和 recommended_courses 字段是否完整。")

    return templates.TemplateResponse(
        request,
        "resume_match.html",
        {
            "title": "课程-岗位智能映射",
            "username": request.session.get("username"),
            "result": result,
            "errors": errors,
            "warnings": warnings,
            "input_data": input_data
        }
    )


@app.get("/interview/mock")
def mock_interview_page(request: Request):
    """
    AI 模拟面试页面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return templates.TemplateResponse(
        request,
        "mock_interview.html",
        {
            "title": "AI 模拟面试",
            "username": request.session.get("username")
        }
    )


MOCK_INTERVIEW_EXAM_CACHE: dict[str, dict] = {}
MOCK_INTERVIEW_LAUNCH_STATUS: dict[str, dict] = {}
MOCK_INTERVIEW_EXAM_TTL_SECONDS = 60 * 60


def cache_mock_interview_exam(user_id: int, context: dict) -> str:
    """短期保存新标签页所需的笔试上下文，避免把简历数据放进 URL。"""
    now = datetime.now().timestamp()
    stale_ids = [
        exam_id
        for exam_id, item in MOCK_INTERVIEW_EXAM_CACHE.items()
        if now - float(item.get("created_at", 0)) > MOCK_INTERVIEW_EXAM_TTL_SECONDS
    ]
    for exam_id in stale_ids:
        MOCK_INTERVIEW_EXAM_CACHE.pop(exam_id, None)

    exam_session = context.get("exam_session") or {}
    exam_id = str(exam_session.get("exam_id") or secrets.token_urlsafe(18))
    MOCK_INTERVIEW_EXAM_CACHE[exam_id] = {
        "user_id": user_id,
        "created_at": now,
        **context,
    }
    return exam_id


@app.get("/interview/session")
def mock_interview_session_redirect(request: Request):
    """无会话 ID 时返回准备页，避免刷新旧 POST 地址出现 405。"""
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return redirect_response
    return RedirectResponse(url="/interview/mock", status_code=303)


@app.get("/interview/session/{exam_id}")
def mock_interview_session_view(request: Request, exam_id: str):
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return redirect_response

    cached = MOCK_INTERVIEW_EXAM_CACHE.get(exam_id)
    user_id = request.session.get("user_id")
    if not cached or cached.get("user_id") != user_id:
        cached = {
            "exam_session": None,
            "uploaded_filename": "",
            "warnings": [],
            "errors": ["该模拟面试会话已过期，请返回准备页重新生成。"],
        }

    return templates.TemplateResponse(
        request,
        "mock_interview_session.html",
        {
            "title": "AI 综合模拟面试",
            "username": request.session.get("username"),
            **{
                key: cached.get(key)
                for key in ("exam_session", "uploaded_filename", "warnings", "errors")
            },
        },
    )


@app.get("/interview/session-loading")
def mock_interview_session_loading(request: Request):
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return redirect_response
    return templates.TemplateResponse(
        request,
        "mock_interview_loading.html",
        {
            "title": "正在生成模拟面试",
            "username": request.session.get("username"),
        },
    )


@app.get("/interview/session/status/{launch_id}")
def mock_interview_session_status(request: Request, launch_id: str):
    user_id = request.session.get("user_id")
    if not user_id:
        return {"status": "error", "message": "登录已失效，请重新登录。"}
    launch = MOCK_INTERVIEW_LAUNCH_STATUS.get(launch_id)
    if not launch:
        return {"status": "pending"}
    if launch.get("user_id") != user_id:
        return {"status": "error", "message": "无权访问该模拟面试。"}
    return {
        "status": launch.get("status", "pending"),
        "redirect_url": launch.get("redirect_url", ""),
        "message": launch.get("message", ""),
    }


async def build_mock_interview_exam_context(
    *,
    target_role: str,
    job_description: str,
    resume_text: str,
    resume_file: UploadFile | None,
) -> dict:
    upload_warnings: list[str] = []
    uploaded_filename = ""
    uploaded_resume_text = ""
    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = await resume_file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content,
        )

    final_target_role = target_role.strip()
    final_job_description = job_description.strip()
    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()
    errors: list[str] = []
    exam_session = None
    if not final_target_role and not final_job_description and not final_resume_text:
        errors.append("请至少填写目标岗位信息，或粘贴、上传一份简历。")
    else:
        try:
            exam_session = await run_in_threadpool(
                build_written_exam,
                final_target_role,
                final_job_description,
                final_resume_text,
            )
        except LLMCallError:
            errors.append("笔试题生成失败，请检查模型配置后重新尝试。")

    return {
        "exam_session": exam_session,
        "uploaded_filename": uploaded_filename,
        "warnings": upload_warnings,
        "errors": errors,
    }


@app.post("/interview/session/create")
async def mock_interview_session_create(
    request: Request,
    launch_id: str = Form(""),
    target_role: str = Form(""),
    job_description: str = Form(""),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
):
    """后台生成笔试题；前端已提前打开加载页，因此无需等待空白标签页。"""
    if not request.session.get("user_id"):
        return {"ok": False, "errors": ["请先登录后使用 AI 模拟面试。"]}

    user_id = int(request.session["user_id"])
    launch_id = launch_id.strip() or secrets.token_urlsafe(18)
    now = datetime.now().timestamp()
    stale_launch_ids = [
        item_id
        for item_id, item in MOCK_INTERVIEW_LAUNCH_STATUS.items()
        if now - float(item.get("created_at", 0)) > MOCK_INTERVIEW_EXAM_TTL_SECONDS
    ]
    for item_id in stale_launch_ids:
        MOCK_INTERVIEW_LAUNCH_STATUS.pop(item_id, None)
    MOCK_INTERVIEW_LAUNCH_STATUS[launch_id] = {
        "user_id": user_id,
        "status": "pending",
        "created_at": now,
    }
    context = await build_mock_interview_exam_context(
        target_role=target_role,
        job_description=job_description,
        resume_text=resume_text,
        resume_file=resume_file,
    )
    exam_id = cache_mock_interview_exam(user_id, context)
    redirect_url = f"/interview/session/{exam_id}"
    MOCK_INTERVIEW_LAUNCH_STATUS[launch_id] = {
        "user_id": user_id,
        "status": "ready",
        "redirect_url": redirect_url,
        "created_at": datetime.now().timestamp(),
    }
    return {
        "ok": not context["errors"],
        "redirect_url": redirect_url,
        "errors": context["errors"],
    }


@app.post("/interview/session")
async def mock_interview_session_page(
    request: Request,
    target_role: str = Form(""),
    job_description: str = Form(""),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
):
    """在新页面中创建岗位笔试，笔试完成后再进入视频面试。"""
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return redirect_response

    context = await build_mock_interview_exam_context(
        target_role=target_role,
        job_description=job_description,
        resume_text=resume_text,
        resume_file=resume_file,
    )
    exam_id = cache_mock_interview_exam(int(request.session["user_id"]), context)
    return RedirectResponse(url=f"/interview/session/{exam_id}", status_code=303)


@app.post("/interview/written/submit")
async def mock_interview_written_submit(request: Request):
    """批改六题笔试，并生成后续视频面试会话。"""
    if not request.session.get("user_id"):
        return {
            "ok": False,
            "errors": ["请先登录后使用 AI 模拟面试。"],
        }

    try:
        payload = await request.json()
    except Exception:
        return {
            "ok": False,
            "errors": ["笔试数据格式不正确，请重新开始。"],
        }

    exam_session = payload.get("exam_session") or {}
    answers = payload.get("answers") or {}
    if not exam_session:
        return {
            "ok": False,
            "errors": ["笔试会话已失效，请返回准备页重新生成。"],
        }

    try:
        written_result = grade_written_exam(exam_session, answers)
        interview_session = build_interview_session(
            target_role=str(exam_session.get("target_role") or "").strip(),
            job_description=str(exam_session.get("job_description") or "").strip(),
            resume_text=str(exam_session.get("resume_text") or "").strip(),
        )
    except ValueError:
        return {
            "ok": False,
            "errors": ["笔试题数据不完整，请返回准备页重新生成。"],
        }
    except LLMCallError:
        return {
            "ok": False,
            "errors": ["视频面试问题生成失败，请稍后重试。"],
        }

    return {
        "ok": True,
        "written_result": written_result,
        "session": interview_session,
        "opening_message": interview_session["opening_message"],
        "question": interview_session["current_question"],
        "round_index": interview_session["current_round"],
        "total_rounds": interview_session["total_rounds"],
    }


@app.post("/interview/start")
async def mock_interview_start(
    request: Request,
    target_role: str = Form(""),
    job_description: str = Form(""),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None)
):
    """
    创建一次模拟面试会话，返回首轮问题。
    """
    if not request.session.get("user_id"):
        return {
            "ok": False,
            "errors": ["请先登录后使用 AI 模拟面试。"]
        }

    upload_warnings = []
    uploaded_filename = ""
    uploaded_resume_text = ""

    if resume_file is not None and resume_file.filename:
        uploaded_filename = resume_file.filename
        file_content = await resume_file.read()
        uploaded_resume_text, upload_warnings = extract_resume_text_from_upload(
            uploaded_filename,
            file_content
        )

    final_target_role = target_role.strip()
    final_job_description = job_description.strip()
    final_resume_text = resume_text.strip() or uploaded_resume_text.strip()

    errors = []
    if not final_target_role and not final_job_description and not final_resume_text:
        errors.append("请至少填写目标岗位信息，或粘贴/上传一份简历。")

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "warnings": upload_warnings
        }

    try:
        session = build_interview_session(
            target_role=final_target_role,
            job_description=final_job_description,
            resume_text=final_resume_text
        )
    except LLMCallError:
        return {
            "ok": False,
            "errors": ["调用LLM失败"],
            "warnings": upload_warnings
        }

    return {
        "ok": True,
        "warnings": upload_warnings,
        "uploaded_filename": uploaded_filename,
        "session": session,
        "opening_message": session["opening_message"],
        "question": session["current_question"],
        "round_index": session["current_round"],
        "total_rounds": session["total_rounds"]
    }


@app.post("/interview/answer")
async def mock_interview_answer(request: Request):
    """
    接收用户回答，返回本轮点评和下一轮问题。
    """
    if not request.session.get("user_id"):
        return {
            "ok": False,
            "errors": ["请先登录后使用 AI 模拟面试。"]
        }

    try:
        payload = await request.json()
    except Exception:
        return {
            "ok": False,
            "errors": ["回答数据格式不正确，请刷新页面后重试。"]
        }

    session = payload.get("session") or {}
    question = (payload.get("question") or session.get("current_question") or "").strip()
    answer = (payload.get("answer") or "").strip()
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []

    try:
        round_index = int(payload.get("round_index") or session.get("current_round") or 1)
    except (TypeError, ValueError):
        round_index = 1

    errors = []
    if not session:
        errors.append("面试会话已失效，请重新进入模拟面试。")
    if not question:
        errors.append("当前问题为空，请重新进入模拟面试。")
    if not answer:
        errors.append("请先输入本轮回答。")

    if errors:
        return {
            "ok": False,
            "errors": errors
        }

    try:
        result = respond_to_interview_answer(
            session=session,
            question=question,
            answer=answer,
            round_index=round_index,
            history=history
        )
    except LLMCallError:
        return {
            "ok": False,
            "errors": ["调用LLM失败"]
        }

    return {
        "ok": True,
        **result
    }


@app.get("/student/input")
def student_input(request: Request):
    """
    学生信息输入页面：登录后才能访问。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return templates.TemplateResponse(
        request,
        "student_input.html",
        {
            "title": "学生信息输入",
            "username": request.session.get("username"),
            "error": ""
        }
    )


@app.post("/student/submit")
def student_submit(
    request: Request,
    name: str = Form(...),
    major: str = Form(...),
    grade: str = Form(...),
    target_job: str = Form(...),
    skills: str = Form(...),
    projects: str = Form(""),
    competitions: str = Form(""),
    certificates: str = Form(""),
    self_intro: str = Form(""),
    db: Session = Depends(get_db)
):
    """
    接收学生提交数据，并保存到 MySQL 数据库。
    每提交一次，就新增一条历史诊断记录。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    student_data = {
        "name": name,
        "major": major,
        "grade": grade,
        "target_job": target_job,
        "skills": skills,
        "projects": projects,
        "competitions": competitions,
        "certificates": certificates,
        "self_intro": self_intro
    }

    try:
        agent_result = run_diagnosis_agent(student_data)
    except LLMCallError:
        return templates.TemplateResponse(
            request,
            "student_input.html",
            {
                "title": "学生信息输入",
                "username": request.session.get("username"),
                "error": "调用LLM失败"
            }
        )

    ability_scores = agent_result["ability_scores"]

    record = DiagnosisRecord(
        user_id=request.session.get("user_id"),
        name=name,
        major=major,
        grade=grade,
        target_job=target_job,
        skills=skills,
        projects=projects,
        competitions=competitions,
        certificates=certificates,
        self_intro=self_intro,
        professional_score=ability_scores["professional"],
        practice_score=ability_scores["practice"],
        tools_score=ability_scores["tools"],
        career_score=ability_scores["career"],
        agent_status="completed",
        agent_result_json=json.dumps(
            agent_result,
            ensure_ascii=False
        )
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return templates.TemplateResponse(
        request,
        "student_submit.html",
        {
            "title": "学生信息提交成功",
            "student": student_data,
            "record_id": record.id,
            "username": request.session.get("username")
        }
    )


@app.get("/ability/profile")
def ability_profile(
    request: Request,
):
    """
    打开一张新的空白能力画像，由用户上传简历后主动开始生成。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    return render_ability_profile(
        request,
        None,
        generation_mode=True,
    )


def _stream_json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _generate_ability_profile_stream(
    *,
    user_id: int,
    message: str,
    uploaded_filename: str,
    resume_text: str,
    upload_warnings: list[str],
    resume_hash: str,
):
    """按简历解析和 LangGraph 节点顺序输出 NDJSON 事件。"""
    db = SessionLocal()
    try:
        yield _stream_json_line({
            "type": "accepted",
            "title": "简历已接收",
            "text": f"已读取 {uploaded_filename}，正在提取学生基础信息。",
            "warnings": upload_warnings,
        })

        student_data = extract_student_profile_from_resume(message, resume_text)
        student_context = {
            **student_data,
            "resume_text": resume_text,
            "normalized_text": resume_text,
        }
        yield _stream_json_line({
            "type": "profile",
            "title": "基础信息已识别",
            "text": (
                f"姓名：{student_data['name']}；专业：{student_data['major']}；"
                f"学历：{student_data['grade']}；目标岗位：{student_data['target_job']}。"
            ),
            "student": {
                key: student_data[key]
                for key in ("name", "major", "grade", "target_job")
            },
        })

        agent_result = None
        for event in run_diagnosis_agent_stream(student_context):
            if event["type"] == "complete":
                agent_result = event["result"]
                continue

            output = event.get("output", {})
            workflow_steps = output.get("workflow_steps") or []
            step = workflow_steps[-1] if workflow_steps else {}
            yield _stream_json_line({
                "type": "agent_step",
                "node": event.get("node", ""),
                "step": step.get("step", ""),
                "title": step.get("agent", "能力画像智能体"),
                "text": step.get("output", "本步骤已完成。"),
                "status": step.get("status", "completed"),
                "ability_scores": output.get("ability_scores"),
                "summary": output.get("summary", ""),
                "data": {
                    key: output[key]
                    for key in (
                        "ability_scores",
                        "score_evidence",
                        "recognized_skills",
                        "profile_tags",
                        "risk_flags",
                        "evidence_cards",
                        "summary",
                        "advantages",
                        "weaknesses",
                        "dimension_insights",
                        "development_focus",
                        "quality_review",
                        "tool_calls",
                        "collaboration_log",
                        "review_findings",
                        "llm_agents",
                    )
                    if key in output
                },
            })

        if not agent_result:
            raise RuntimeError("能力画像工作流未返回最终结果")

        agent_result["student_profile"] = {
            key: student_data[key]
            for key in ("name", "major", "grade", "target_job")
        }
        agent_result = attach_resume_cache_metadata(agent_result, resume_hash)
        ability_scores = agent_result["ability_scores"]
        record = DiagnosisRecord(
            user_id=user_id,
            name=student_data["name"],
            major=student_data["major"],
            grade=student_data["grade"],
            target_job=student_data["target_job"],
            skills=student_data["skills"],
            projects=student_data["projects"],
            competitions=student_data["competitions"],
            certificates=student_data["certificates"],
            self_intro=student_data["self_intro"],
            professional_score=ability_scores["professional"],
            practice_score=ability_scores["practice"],
            tools_score=ability_scores["tools"],
            career_score=ability_scores["career"],
            agent_status="completed",
            agent_result_json=json.dumps(agent_result, ensure_ascii=False),
            resume_hash=resume_hash,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        yield _stream_json_line({
            "type": "complete",
            "title": "能力画像生成完成",
            "text": agent_result.get("summary", "四维能力画像已经生成。"),
            "redirect_url": f"/ability/profile/{record.id}",
            "metrics": {
                "agent_roster": len(agent_result.get("agent_roster", [])),
                "llm_agents": len(agent_result.get("llm_agents", [])),
                "tool_calls": len(agent_result.get("tool_calls", [])),
                "collaboration_log": len(agent_result.get("collaboration_log", [])),
            },
        })
    except LLMCallError:
        db.rollback()
        yield _stream_json_line({
            "type": "error",
            "title": "生成失败",
            "text": "调用 LLM 失败，请检查模型配置后重试。",
        })
    except Exception:
        db.rollback()
        logger.exception("流式生成能力画像失败")
        yield _stream_json_line({
            "type": "error",
            "title": "生成失败",
            "text": "能力画像生成失败，请稍后重试。",
        })
    finally:
        db.close()


def _find_cached_ability_profile(
    db: Session,
    *,
    user_id: int,
    resume_hash: str,
) -> dict | None:
    """从当前用户最近的诊断记录中查找同一份简历的完整画像。"""
    record = db.scalar(
        select(DiagnosisRecord)
        .where(
            DiagnosisRecord.user_id == user_id,
            DiagnosisRecord.resume_hash == resume_hash,
            DiagnosisRecord.agent_status == "completed",
            DiagnosisRecord.agent_result_json.is_not(None),
        )
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .limit(1)
    )

    if record is not None:
        agent_result = load_agent_result(record)
        if is_matching_cached_result(agent_result, resume_hash):
            return {
                "record_id": record.id,
                "student": build_student_data(record),
                "agent_result": agent_result,
            }
    return None


def _generate_cached_ability_profile_stream(
    *,
    cached_profile: dict,
    uploaded_filename: str,
):
    """把 diagnosis_records 中的完整画像重新拆段，保持逐字生成体验。"""
    record_id = cached_profile["record_id"]
    student_data = cached_profile["student"]
    agent_result = cached_profile["agent_result"]

    yield _stream_json_line({
        "type": "accepted",
        "cache_hit": True,
        "title": "已命中历史画像缓存",
        "text": f"检测到 {uploaded_filename} 与历史简历相同，正在读取数据库画像。",
    })
    yield _stream_json_line({
        "type": "profile",
        "cache_hit": True,
        "title": "基础信息已从缓存读取",
        "text": "已从 diagnosis_records 读取学生基础信息。",
        "student": {
            key: student_data.get(key, "无")
            for key in ("name", "major", "grade", "target_job")
        },
    })

    for event in build_cached_agent_events(agent_result):
        yield _stream_json_line(event)

    yield _stream_json_line({
        "type": "complete",
        "cache_hit": True,
        "title": "缓存画像加载完成",
        "text": agent_result.get("summary", "四维能力画像已经加载。"),
        "redirect_url": f"/ability/profile/{record_id}",
        "metrics": {
            "agent_roster": len(agent_result.get("agent_roster", [])),
            "llm_agents": len(agent_result.get("llm_agents", [])),
            "tool_calls": len(agent_result.get("tool_calls", [])),
            "collaboration_log": len(agent_result.get("collaboration_log", [])),
        },
    })


@app.post("/ability/profile/generate")
async def ability_profile_generate(
    request: Request,
    resume_file: UploadFile | None = File(None),
):
    """接收一份简历，并按智能体执行顺序实时返回画像文本。"""
    user_id = request.session.get("user_id")
    if not user_id:
        return StreamingResponse(
            iter([_stream_json_line({
                "type": "error",
                "title": "请先登录",
                "text": "登录后才能生成能力画像。",
            })]),
            media_type="application/x-ndjson",
        )

    if resume_file is None or not resume_file.filename:
        return StreamingResponse(
            iter([_stream_json_line({
                "type": "error",
                "title": "缺少简历",
                "text": "请先拖入 PDF、Word 或文本简历。",
            })]),
            media_type="application/x-ndjson",
        )

    uploaded_filename = resume_file.filename
    file_content = await resume_file.read()
    resume_text, upload_warnings = extract_resume_text_from_upload(
        uploaded_filename,
        file_content,
    )
    if not resume_text:
        message = upload_warnings[0] if upload_warnings else "未能从简历中读取有效文本。"
        return StreamingResponse(
            iter([_stream_json_line({
                "type": "error",
                "title": "简历解析失败",
                "text": message,
            })]),
            media_type="application/x-ndjson",
        )

    resume_hash = build_resume_cache_hash(resume_text)
    cache_db = SessionLocal()
    try:
        cached_profile = _find_cached_ability_profile(
            cache_db,
            user_id=user_id,
            resume_hash=resume_hash,
        )
    finally:
        cache_db.close()

    if cached_profile:
        response = StreamingResponse(
            _generate_cached_ability_profile_stream(
                cached_profile=cached_profile,
                uploaded_filename=uploaded_filename,
            ),
            media_type="application/x-ndjson",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["X-Ability-Profile-Cache"] = "HIT"
        return response

    response = StreamingResponse(
        _generate_ability_profile_stream(
            user_id=user_id,
            message="请根据这份简历生成能力画像",
            uploaded_filename=uploaded_filename,
            resume_text=resume_text,
            upload_warnings=upload_warnings,
            resume_hash=resume_hash,
        ),
        media_type="application/x-ndjson",
    )
    response.headers["Cache-Control"] = "no-cache, no-transform"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-Ability-Profile-Cache"] = "MISS"
    return response


@app.get("/ability/profile/{record_id}")
def ability_profile_detail(
    record_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    根据历史记录编号查看指定的一次能力画像。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    record = db.scalar(
        select(DiagnosisRecord)
        .where(
            DiagnosisRecord.id == record_id,
            DiagnosisRecord.user_id == request.session.get("user_id"),
        )
    )

    if record is None:
        raise HTTPException(status_code=404, detail="该诊断记录不存在")

    return render_ability_profile(request, record)


@app.post("/ability/profile/{record_id}/export-json")
def ability_profile_export_json(
    record_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    将本次能力画像按结构化 JSON 输出到当前用户桌面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    record = db.scalar(
        select(DiagnosisRecord)
        .where(
            DiagnosisRecord.id == record_id,
            DiagnosisRecord.user_id == request.session.get("user_id"),
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="该诊断记录不存在")

    page_path = f"/ability/profile/{record.id}"
    try:
        payload = build_ability_profile_export_payload(db, record)
        file_path = write_ability_profile_json_to_desktop(
            payload,
            record.name,
        )
    except Exception as exc:
        return build_export_redirect(
            page_path,
            error=f"JSON 导出失败：{exc}",
        )

    return build_export_redirect(
        page_path,
        message=f"JSON 文件已输出到桌面：{file_path.name}",
    )


@app.get("/history")
def history_records(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    历史诊断记录页面。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    records = db.scalars(
        select(DiagnosisRecord)
        .order_by(
            DiagnosisRecord.created_at.desc(),
            DiagnosisRecord.id.desc()
        )
    ).all()

    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "title": "历史诊断记录",
            "records": records,
            "username": request.session.get("username")
        }
    )


@app.get("/job/match")
def job_match(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    岗位匹配页面快速入口。

    只读取已保存画像并执行/读取本地匹配，不在页面入口调用任何 LLM。
    如果用户此前完成过 AI 精排，则优先显示持久化的精排缓存。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    user_id = request.session.get("user_id")

    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(
            DiagnosisRecord.created_at.desc(),
            DiagnosisRecord.id.desc()
        )
        .first()
    )

    if student_record is None:
        student_data = {
            "name": "无",
            "major": "无",
            "grade": "无",
            "target_job": "无",
            "skills": "无",
            "projects": "无",
            "competitions": "无",
            "certificates": "无",
            "self_intro": "无"
        }

        job_matches = []
        match_source = "local"
        match_cached = False

    else:
        student_data = build_student_data(student_record)
        job_matches, match_source, match_cached = get_job_matches_for_record(
            db,
            student_record,
            top_n=10,
        )

    return templates.TemplateResponse(
        request,
        "job_match.html",
        {
            "title": "岗位匹配结果",
            "student": student_data,
            "job_matches": job_matches,
            "match_source": match_source,
            "match_cached": match_cached,
            "username": request.session.get("username"),
            "csrf_token": get_or_create_csrf_token(request),
            "error": "",
            "export_message": request.query_params.get("export_message", ""),
            "export_error": request.query_params.get("export_error", ""),
        }
    )


@app.get("/job/match/companies")
def job_match_companies(
    request: Request,
    job_name: str,
    db: Session = Depends(get_db)
):
    """
    为 TOP5 岗位查询数据库中的 TOP5 公司。
    使用与岗位精排一致的双向分数结构：学生适岗分 + 岗位适生分。
    """
    if not request.session.get("user_id"):
        return {"ok": False, "error": "请先登录。"}

    user_id = request.session.get("user_id")
    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .first()
    )
    if student_record is None:
        return {"ok": False, "error": "请先完成学生能力诊断，再查看岗位公司。"}

    target_job_name = str(job_name or "").strip()
    if not target_job_name:
        return {"ok": False, "error": "岗位名称不能为空。"}

    companies = build_top_company_matches_for_job(
        db,
        student_record,
        target_job_name,
        top_n=5,
    )
    return {
        "ok": True,
        "job_name": target_job_name,
        "match_source": "database_bidirectional",
        "companies": companies,
        "warning": "" if companies else "数据库中暂未筛选到该岗位对应的公司，请先导入带公司字段的岗位数据。",
    }


@app.post("/job/match/company/apply")
async def job_match_company_apply(
    request: Request,
    job_record_id: int = Form(...),
    company_name: str = Form(...),
    job_name: str = Form(...),
    privacy_consent: str = Form(""),
    csrf_token: str = Form(""),
    resume_file: UploadFile | None = File(None),
    db: Session = Depends(get_db)
):
    """
    从 TOP5 公司列表投递 PDF 简历。
    投递记录进入学生-学校-企业三方协同表，由学校先审核真实性。
    """
    if not request.session.get("user_id"):
        return {"ok": False, "errors": ["请先登录。"]}

    if get_session_role(request) != "student":
        return {"ok": False, "errors": ["请以学生身份登录后再投递简历。"]}

    try:
        require_valid_csrf(request, csrf_token)
        enforce_collaboration_rate_limit(request, "company-apply", limit=5, window_seconds=600)
    except HTTPException as exc:
        return {"ok": False, "errors": [str(exc.detail)]}

    if privacy_consent != "accepted":
        return {"ok": False, "errors": ["请先同意学校核验原始材料并向目标企业转交脱敏简历。"]}

    if resume_file is None or not resume_file.filename:
        return {"ok": False, "errors": ["请先拖入 PDF 简历。"]}

    if Path(resume_file.filename).suffix.lower() != ".pdf":
        return {"ok": False, "errors": ["当前投递入口仅支持 PDF 简历。"]}

    user_id = request.session.get("user_id")
    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .first()
    )
    if student_record is None:
        return {"ok": False, "errors": ["请先完成学生能力诊断，再投递简历。"]}

    file_content = await resume_file.read()
    if len(file_content) > 5 * 1024 * 1024:
        return {"ok": False, "errors": ["PDF 简历不能超过 5MB。"]}
    resume_text, upload_warnings = extract_resume_text_from_upload(
        resume_file.filename,
        file_content,
    )
    if not resume_text.strip():
        return {
            "ok": False,
            "errors": upload_warnings or ["PDF 简历未解析出有效文本，请更换文件后重试。"],
        }

    company_record = db.get(JobKnowledgeRecord, job_record_id)
    final_company_name = (
        (company_record.company_name if company_record else "")
        or str(company_name or "").strip()
    )
    final_job_name = (
        (company_record.job_name if company_record else "")
        or str(job_name or "").strip()
    )
    if not final_company_name or not final_job_name:
        return {"ok": False, "errors": ["公司或岗位信息缺失，请刷新页面后重试。"]}

    application = TriPartyResumeApplicationRecord(
        student_user_id=user_id,
        student_username=request.session.get("username", ""),
        student_name=student_record.name,
        major=student_record.major,
        target_company=final_company_name,
        target_job=final_job_name,
        resume_text=resume_text.strip(),
        resume_hash=hashlib.sha256(resume_text.encode("utf-8")).hexdigest(),
        status="school_review",
        school_feedback="由 TOP5 公司投递入口提交，等待学校老师审核简历真实性。",
    )
    db.add(application)
    db.commit()

    return {
        "ok": True,
        "application_id": application.id,
        "message": f"PDF 简历已投递至 {final_company_name}，当前进入学校端真实性审核，审核通过后将自动转企业。",
        "warnings": upload_warnings,
    }


@app.post("/job/match/export-pdf")
def job_match_export_pdf(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    将 TOP5 岗位匹配结果和对应路径规划输出为 PDF 到桌面。
    """
    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    user_id = request.session.get("user_id")
    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .first()
    )
    if student_record is None:
        return build_export_redirect(
            "/job/match",
            error="请先完成学生能力诊断，再导出 TOP5 岗位报告。",
        )

    try:
        student_data = build_student_data(student_record)
        job_matches, _, _ = get_job_matches_for_record(
            db,
            student_record,
            top_n=10,
        )
        top5_matches = ensure_gap_paths_for_job_matches(
            db,
            student_record,
            job_matches[:5],
        )
        if not top5_matches:
            return build_export_redirect(
                "/job/match",
                error="暂无 TOP5 岗位推荐，请先生成岗位匹配结果。",
            )
        file_path = write_job_match_pdf_to_desktop(
            student_data,
            top5_matches,
        )
    except Exception as exc:
        return build_export_redirect(
            "/job/match",
            error=f"PDF 导出失败：{exc}",
        )

    return build_export_redirect(
        "/job/match",
        message=f"PDF 文件已输出到桌面：{file_path.name}",
    )


@app.post("/job/match/refine")
def job_match_refine(
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """按需使用一次 LLM 对本地 TOP10 精排，失败时保留本地结果。"""
    redirect_response = get_login_redirect(request)
    if redirect_response:
        return {"ok": False, "error": "请先登录。"}

    user_id = request.session.get("user_id")
    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .first()
    )
    if student_record is None:
        return {"ok": False, "error": "请先完成学生能力诊断。"}

    student_data = build_student_data(student_record)
    assessment = build_match_assessment(student_record)
    job_records = db.query(JobKnowledgeRecord).all()
    job_version = build_job_version(job_records)

    cached = load_persistent_match_cache(
        db,
        student_record,
        job_version,
        "llm",
    )
    if cached is not None:
        response.headers["X-Job-Match-Cache"] = "HIT"
        return {
            "ok": True,
            "cached": True,
            "source": "llm",
            "job_matches": cached[:5],
        }

    local_matches = load_persistent_match_cache(
        db,
        student_record,
        job_version,
        "local",
    )
    if local_matches is None:
        local_matches = calculate_job_match(
            student_data,
            job_records,
            assessment=assessment,
            top_n=10,
        )
        save_persistent_match_cache(
            db,
            student_record,
            job_version,
            "local",
            local_matches,
        )

    try:
        refined = refine_job_matches_with_llm(
            student_data=student_data,
            local_matches=local_matches,
            assessment=assessment,
            top_n=10,
        )
    except Exception:
        refined = local_matches

    used_llm = any(bool(item.get("used_llm")) for item in refined)
    if used_llm:
        save_persistent_match_cache(
            db,
            student_record,
            job_version,
            "llm",
            refined,
        )

    response.headers["X-Job-Match-Cache"] = "MISS"
    return {
        "ok": used_llm,
        "cached": False,
        "source": "llm" if used_llm else "local",
        "job_matches": refined[:5],
        "warning": "" if used_llm else "AI 精排超时或失败，已保留本地匹配结果。",
    }


@app.post("/job/match/path")
async def job_match_path(
    request: Request,
    db: Session = Depends(get_db)
):
    """用户点击岗位后，才为该单个岗位生成并持久化成长路径。"""
    if not request.session.get("user_id"):
        return {"ok": False, "error": "请先登录。"}

    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "error": "请求数据格式不正确。"}

    job_name = str(payload.get("job_name") or "").strip()
    if not job_name:
        return {"ok": False, "error": "岗位名称不能为空。"}

    user_id = request.session.get("user_id")
    student_record = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(DiagnosisRecord.created_at.desc(), DiagnosisRecord.id.desc())
        .first()
    )
    if student_record is None:
        return {"ok": False, "error": "请先完成学生能力诊断。"}

    agent_result = load_agent_result(student_record)
    cached_paths = agent_result.get("top5_gap_paths", [])
    if isinstance(cached_paths, dict):
        cached_paths = cached_paths.get("top5_gap_paths", [])
    if not isinstance(cached_paths, list):
        cached_paths = []

    for item in cached_paths:
        if isinstance(item, dict) and item.get("job_name") == job_name:
            return {
                "ok": True,
                "cached": True,
                "used_llm": bool(item.get("used_llm")),
                "path": item,
            }

    student_data = build_student_data(student_record)
    job_records = db.query(JobKnowledgeRecord).all()
    job_version = build_job_version(job_records)
    matches = (
        load_persistent_match_cache(db, student_record, job_version, "llm")
        or load_persistent_match_cache(db, student_record, job_version, "local")
    )
    if matches is None:
        matches = calculate_job_match(
            student_data,
            job_records,
            assessment=build_match_assessment(student_record),
            top_n=10,
        )
        save_persistent_match_cache(
            db,
            student_record,
            job_version,
            "local",
            matches,
        )

    selected_job = next(
        (item for item in matches if item.get("job_name") == job_name),
        None,
    )
    if selected_job is None:
        return {"ok": False, "error": "当前匹配结果中未找到该岗位。"}

    path_result = generate_top5_gap_paths(
        student_data=student_data,
        job_recommendations=[selected_job],
        use_llm=True,
    )
    generated = path_result.get("top5_gap_paths", [])
    if not generated:
        return {"ok": False, "error": "成长路径生成失败，请稍后重试。"}

    path = generated[0]
    path["used_llm"] = bool(path_result.get("used_llm"))
    cached_paths.append(path)
    agent_result["top5_gap_paths"] = cached_paths
    agent_result["path_agent_warning"] = path_result.get("agent_warning", "")
    student_record.agent_result_json = json.dumps(agent_result, ensure_ascii=False)
    db.add(student_record)
    db.commit()

    return {
        "ok": True,
        "cached": False,
        "used_llm": bool(path_result.get("used_llm")),
        "warning": path_result.get("agent_warning", ""),
        "path": path,
    }


@app.get("/health")
def health_check():
    """
    健康检查接口
    """
    return {
        "status": "ok",
        "message": "系统运行正常"
    }
@app.get("/growth/trend")
def growth_trend(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    当前登录用户的个人全部成长轨迹。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    user_id = request.session.get("user_id")

    records = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user_id)
        .order_by(
            DiagnosisRecord.created_at.asc(),
            DiagnosisRecord.id.asc()
        )
        .all()
    )

    if len(records) == 0:
        return templates.TemplateResponse(
            request,
            "growth_trend.html",
            {
                "title": "个人成长轨迹",
                "has_data": False,
                "message": "当前账号暂无诊断记录，请先完成一次学生能力诊断。",
                "username": request.session.get("username")
            }
        )

    trend = build_growth_trend(records)

    return templates.TemplateResponse(
        request,
        "growth_trend.html",
        {
            "title": "个人成长轨迹",
            "has_data": True,
            "trend": trend,
            "username": request.session.get("username")
        }
    )
