import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, String, Text, Integer, DateTime, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)
from app.agent.diagnosis_agent import run_diagnosis_agent
from app.services.job_match_service import calculate_job_match

# =========================================================
# 项目路径配置
# =========================================================

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"
# data 目录
DATA_DIR = BASE_DIR / "data"

# 最新一次提交的学生数据文件
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
    echo=True,              # 开发阶段显示 SQL 语句，后面可以改成 False
    pool_pre_ping=True      # 自动检测数据库连接是否有效
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
    用数据库保存岗位与技能、项目、课程、证书之间的关系
    """
    __tablename__ = "job_knowledge_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    job_name: Mapped[str] = mapped_column(String(100), nullable=False)

    required_skills_json: Mapped[str] = mapped_column(Text, nullable=False)
    related_projects_json: Mapped[str] = mapped_column(Text, default="[]")
    recommended_courses_json: Mapped[str] = mapped_column(Text, default="[]")
    recommended_certificates_json: Mapped[str] = mapped_column(Text, default="[]")

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False
    )

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
                "recommended_courses": ["Python程序设计", "数据库原理", "人工智能导论"],
                "recommended_certificates": ["英语六级", "软考程序员"]
            },
            {
                "job_name": "Java后端工程师",
                "required_skills": ["Java", "Spring Boot", "MySQL", "Redis", "Linux"],
                "related_projects": ["学生管理系统", "电商后台系统", "权限管理系统"],
                "recommended_courses": ["Java程序设计", "数据库原理", "软件工程"],
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
                required_skills_json=json.dumps(job["required_skills"], ensure_ascii=False),
                related_projects_json=json.dumps(job["related_projects"], ensure_ascii=False),
                recommended_courses_json=json.dumps(job["recommended_courses"], ensure_ascii=False),
                recommended_certificates_json=json.dumps(job["recommended_certificates"], ensure_ascii=False)
            )

            db.add(record)

        db.commit()
        print("[JobKnowledge] 岗位知识图谱初始化完成")

    finally:
        db.close()
# 自动创建数据库表
# 如果 diagnosis_records 表不存在，程序启动时会自动创建
Base.metadata.create_all(bind=engine)

init_job_knowledge_data()


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

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


# =========================================================
# 模拟能力评分函数
# =========================================================

def calculate_ability_scores(student_data: dict) -> dict:
    """
    当前仍然使用模拟能力分数。
    后续你可以在这里接入真正的岗位能力诊断算法。
    """
    return {
        "professional": 75,
        "practice": 70,
        "tools": 80,
        "career": 65
    }


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
            "name": "未填写",
            "major": "未填写",
            "grade": "未填写",
            "target_job": "未填写",
            "skills": "",
            "projects": "",
            "competitions": "",
            "certificates": "",
            "self_intro": ""
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

    agent_result = load_agent_result(record)

    return templates.TemplateResponse(
        request,
        "ability_profile.html",
        {
            "title": "学生能力画像",
            "student": student_data,
            "ability_scores": ability_scores,
            "ability_explain": ability_explain,
            "record": record,

            "agent_result": agent_result,
            "summary": agent_result.get("summary", ""),
            "advantages": agent_result.get("advantages", []),
            "weaknesses": agent_result.get("weaknesses", []),
            "job_match_analysis": agent_result.get("job_match_analysis", ""),
            "job_recommendations": agent_result.get("job_recommendations", []),
            "growth_path": agent_result.get("growth_path", [])
        }
    )
#读取智能体 JSON 结果
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

# =========================================================
# 页面路由
# =========================================================

@app.get("/")
def index(request: Request):
    """
    系统首页
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "岗位能力达成学生成长诊断与精准就业智能体系统"
        }
    )


@app.get("/student/input")
def student_input(request: Request):
    """
    学生信息输入页面
    """
    return templates.TemplateResponse(
        request,
        "student_input.html",
        {
            "title": "学生信息输入"
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

    agent_result = run_diagnosis_agent(student_data)

    ability_scores = agent_result["ability_scores"]

    record = DiagnosisRecord(
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
        agent_status = "completed",
        agent_result_json = json.dumps(
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
            "record_id": record.id
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
            "records": records
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

@app.get("/job/match")
def job_match(request: Request, db: Session = Depends(get_db)):
    """
    岗位匹配页面：
    学生信息从 diagnosis_records 表读取；
    岗位知识图谱从 job_knowledge_records 表读取。
    """

    student_record = (
        db.query(DiagnosisRecord)
        .order_by(DiagnosisRecord.created_at.desc())
        .first()
    )

    if student_record is None:
        student_data = {
            "name": "未填写",
            "major": "未填写",
            "grade": "未填写",
            "target_job": "未填写",
            "skills": "",
            "projects": "",
            "competitions": "",
            "certificates": "",
            "self_intro": ""
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

        job_matches = calculate_job_match(student_data, job_records)

    return templates.TemplateResponse(
        request,
        "job_match.html",
        {
            "title": "岗位匹配结果",
            "student": student_data,
            "job_matches": job_matches
        }
    )