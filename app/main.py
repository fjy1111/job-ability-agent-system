import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import Boolean, Float, ForeignKey, create_engine, String, Text, Integer, DateTime, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)

from app.services.llm_ability_match_service import calculate_job_match
from app.agent.diagnosis_agent import AGENT_ROSTER, run_diagnosis_agent
from app.services.llm_errors import LLMCallError
from app.services.llm_gap_path_agent import generate_top5_gap_paths
from app.services.mock_interview_service import (
    build_interview_session,
    respond_to_interview_answer,
)
from app.services.resume_optimizer_service import (
    extract_resume_text_from_upload,
    optimize_resume,
)
from app.services.course_job_mapping_service import build_course_job_mapping_graph
from app.services.resume_profile_extractor_service import (
    extract_student_profile_from_resume,
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
    注意：当前密码为明文保存，仅适合比赛本地演示。
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


# 自动创建数据库表
Base.metadata.create_all(bind=engine)

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

# Session 中间件：用于保存登录状态
# 如果后续想更规范，可以把 SESSION_SECRET_KEY 放到 .env 中
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "job-ability-agent-system-secret-key")
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# =========================================================
# 登录状态工具函数
# =========================================================

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


# =========================================================
# 业务辅助函数
# =========================================================

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


def detect_agent_intent(message: str) -> str:
    text = (message or "").strip()
    if any(keyword in text for keyword in ["优化", "改简历", "润色", "简历优化"]):
        return "resume"
    if any(keyword in text for keyword in ["画像", "能力", "诊断", "分析我"]):
        return "profile"
    return ""


def render_ability_profile(
    request: Request,
    record: DiagnosisRecord | None
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
            "agent_warning": agent_result.get("agent_warning", "")
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
    处理用户注册。
    当前使用明文密码，仅适合本地比赛演示。
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
        password=password
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "title": "用户登录",
            "error": "",
            "message": "注册成功，请登录"
        }
    )


@app.get("/login")
def login_page(request: Request):
    """
    登录页面
    """

    if request.session.get("user_id"):
        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "title": "用户登录",
            "error": "",
            "message": ""
        }
    )


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    处理用户登录。
    """

    username = username.strip()
    password = password.strip()

    if not username or not password:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "title": "用户登录",
                "error": "用户名和密码不能为空",
                "message": ""
            }
        )

    user = db.scalar(
        select(User).where(User.username == username)
    )

    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "title": "用户登录",
                "error": "用户不存在",
                "message": ""
            }
        )

    if user.password != password:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "title": "用户登录",
                "error": "密码错误",
                "message": ""
            }
        )

    # 登录成功：保存登录状态
    request.session["user_id"] = user.id
    request.session["username"] = user.username

    return RedirectResponse(
        url="/",
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
            "message": ""
        }
    )


@app.post("/agent/chat")
async def agent_chat(
    request: Request,
    message: str = Form(""),
    resume_file: UploadFile | None = File(None),
    db: Session = Depends(get_db)
):
    """
    首页聊天式智能体入口：根据用户指令分支到能力画像或简历优化。
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
            "errors": ["请告诉智能体要生成画像还是优化简历。"],
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

    result = None
    if not errors:
        result = build_course_job_mapping_graph(
            resume_text=final_resume_text,
            job_records=job_records,
            top_jobs_per_course=top_jobs_per_course,
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
            "warnings": upload_warnings,
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
    db: Session = Depends(get_db)
):
    """
    显示最近一次提交的学生能力画像。
    """

    redirect_response = get_login_redirect(request)

    if redirect_response:
        return redirect_response

    latest_record = db.scalar(
        select(DiagnosisRecord)
        .order_by(
            DiagnosisRecord.created_at.desc(),
            DiagnosisRecord.id.desc()
        )
        .limit(1)
    )

    return render_ability_profile(request, latest_record)


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

    record = db.get(DiagnosisRecord, record_id)

    if record is None:
        raise HTTPException(status_code=404, detail="该诊断记录不存在")

    return render_ability_profile(request, record)


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
    岗位匹配页面：
    学生信息从 diagnosis_records 表读取；
    岗位知识图谱从 job_knowledge_records 表读取。
    同时为 TOP5 岗位生成差距清单、补齐路径、推荐项目和学习阶段。
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

    else:
        student_data = {
            "name": student_record.name,
            "major": student_record.major,
            "grade": student_record.grade,
            "target_job": student_record.target_job,
            "skills": student_record.skills,
            "projects": student_record.projects,
            "competitions": student_record.competitions,
            "certificates": student_record.certificates,
            "self_intro": student_record.self_intro
        }

        job_records = db.query(JobKnowledgeRecord).all()

        try:
            job_matches = calculate_job_match(student_data, job_records)
        except LLMCallError:
            return templates.TemplateResponse(
                request,
                "job_match.html",
                {
                    "title": "岗位匹配结果",
                    "student": student_data,
                    "job_matches": [],
                    "username": request.session.get("username"),
                    "error": "调用LLM失败"
                }
            )

        # 只分析 TOP5
        top5_jobs = job_matches[:5]

        # 读取原来的 agent_result_json，避免覆盖能力诊断结果
        agent_result = load_agent_result(student_record)

        # 如果之前没有生成过 TOP5 差距路径，就调用大模型生成一次
        top5_gap_paths = agent_result.get("top5_gap_paths", [])
        if isinstance(top5_gap_paths, dict):
            top5_gap_paths = top5_gap_paths.get("top5_gap_paths", [])
        if not isinstance(top5_gap_paths, list):
            top5_gap_paths = []

        if not top5_gap_paths and top5_jobs:
            try:
                gap_path_result = generate_top5_gap_paths(
                    student_data=student_data,
                    job_recommendations=top5_jobs
                )
            except LLMCallError:
                return templates.TemplateResponse(
                    request,
                    "job_match.html",
                    {
                        "title": "岗位匹配结果",
                        "student": student_data,
                        "job_matches": [],
                        "username": request.session.get("username"),
                        "error": "调用LLM失败"
                    }
                )

            top5_gap_paths = gap_path_result.get("top5_gap_paths", [])
            agent_result["top5_gap_paths"] = top5_gap_paths
            agent_result["used_llm"] = gap_path_result.get("used_llm", False)
            agent_result["agent_warning"] = gap_path_result.get("agent_warning", "")

            student_record.agent_result_json = json.dumps(
                agent_result,
                ensure_ascii=False
            )

            db.add(student_record)
            db.commit()
            db.refresh(student_record)

        # 把每个岗位的差距路径挂到对应 job_match 上
        gap_map = {
            item.get("job_name"): item
            for item in top5_gap_paths
            if isinstance(item, dict)
        }

        for index, job in enumerate(job_matches):
            if index < 5:
                job["gap_detail"] = gap_map.get(job.get("job_name")) or {}
            else:
                job["gap_detail"] = {}

    return templates.TemplateResponse(
        request,
        "job_match.html",
        {
            "title": "岗位匹配结果",
            "student": student_data,
            "job_matches": job_matches,
            "username": request.session.get("username"),
            "error": ""
        }
    )


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
