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

from sqlalchemy import create_engine, String, Text, Integer, DateTime, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)

from app.agent.diagnosis_agent import run_diagnosis_agent
from app.services.llm_ability_match_service import calculate_job_match, calculate_ai_job_match, score_four_dimensions_llm
from app.agent.diagnosis_agent import ABILITY_DIMENSIONS, AGENT_ROSTER, run_diagnosis_agent
from app.services.job_match_service import calculate_job_match
from app.services.llm_gap_path_agent import generate_top5_gap_paths
from app.services.mock_interview_service import (
    build_interview_session,
    respond_to_interview_answer,
)
from app.services.resume_optimizer_service import (
    extract_resume_text_from_upload,
    optimize_resume,
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
    echo=True,
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


# 自动创建数据库表
Base.metadata.create_all(bind=engine)

# 初始化岗位知识图谱数据
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


def profile_score_level(score: int) -> str:
    if score >= 85:
        return "优势突出"
    if score >= 70:
        return "稳定具备"
    if score >= 55:
        return "已有基础"
    if score >= 40:
        return "需要补强"
    return "信息不足"


def build_ability_profile_fallback(
    student_data: dict,
    ability_scores: dict,
    agent_result: dict
) -> dict:
    """
    旧记录可能保存了岗位匹配版总结。
    画像页只动态回填能力画像字段，不把旧岗位推荐内容带入页面。
    """

    recognized_skills = agent_result.get("recognized_skills", [])
    score_evidence = agent_result.get("score_evidence", {})

    strongest_key = max(ABILITY_DIMENSIONS, key=lambda key: ability_scores.get(key, 0))
    weakest_key = min(ABILITY_DIMENSIONS, key=lambda key: ability_scores.get(key, 0))
    average_score = round(sum(ability_scores.values()) / max(len(ability_scores), 1), 1)

    dimension_insights = []
    evidence_cards = []

    for key, meta in ABILITY_DIMENSIONS.items():
        score = ability_scores.get(key, 0)
        evidence = score_evidence.get(key, [])
        if not evidence:
            evidence = ["旧记录未保存该维度的详细证据，可重新诊断生成完整证据卡。"]

        level = profile_score_level(score)
        dimension_insights.append({
            "key": key,
            "name": meta["name"],
            "score": score,
            "level": level,
            "conclusion": f"{meta['name']}当前处于“{level}”水平。",
            "evidence": evidence,
            "next_action": meta["next_action"]
        })
        evidence_cards.append({
            "dimension": key,
            "name": meta["name"],
            "agent": meta["agent"],
            "evidence": evidence,
            "confidence": "中" if score >= 55 else "低",
            "interpretation": f"该证据卡根据历史分数和已保存证据动态回填，当前分数为 {score} 分。"
        })

    profile_tags = []
    if recognized_skills:
        profile_tags.append(f"显性技能 {len(recognized_skills)} 项")
    if student_data.get("projects"):
        profile_tags.append("有项目经历")
    else:
        profile_tags.append("项目证据待补充")
    if student_data.get("target_job"):
        profile_tags.append("职业目标已填写")

    risk_flags = []
    if not student_data.get("projects"):
        risk_flags.append("项目经历为空，实践能力缺少可展示证据。")
    if not student_data.get("self_intro"):
        risk_flags.append("自我介绍为空，职业表达和经历复盘证据不足。")
    if ability_scores.get(weakest_key, 0) < 55:
        risk_flags.append(f"{ABILITY_DIMENSIONS[weakest_key]['name']}低于 55 分，需要优先补强。")

    development_focus = []
    for index, key in enumerate(
        sorted(ABILITY_DIMENSIONS, key=lambda item: ability_scores.get(item, 0))[:3],
        start=1
    ):
        meta = ABILITY_DIMENSIONS[key]
        development_focus.append({
            "name": meta["name"],
            "priority": "高" if index == 1 else "中",
            "reason": f"当前得分 {ability_scores.get(key, 0)} 分，处于“{profile_score_level(ability_scores.get(key, 0))}”水平。",
            "action": meta["next_action"]
        })

    growth_path = [
        {
            "stage": "第一阶段：画像校准与基础补强",
            "duration": "第1-2周",
            "goal": f"围绕{ABILITY_DIMENSIONS[weakest_key]['name']}补齐基础证据。",
            "actions": [
                "整理课程、技能、项目、证书清单，补充掌握程度。",
                "针对最低分维度完成一轮知识点复盘。",
                "把经历改写为可验证的能力证据。"
            ],
            "deliverables": ["能力证据清单", "最低分维度补强笔记"]
        },
        {
            "stage": "第二阶段：实践作品沉淀",
            "duration": "第3-6周",
            "goal": "形成一个能支撑能力画像的项目或作品材料。",
            "actions": [
                "选择一个真实场景设计小型项目。",
                "使用 Git 记录开发过程并完善 README。",
                "总结项目中的个人职责、技术动作和结果。"
            ],
            "deliverables": ["项目仓库", "项目复盘文档"]
        },
        {
            "stage": "第三阶段：表达复盘与二次画像",
            "duration": "第7-8周",
            "goal": "把能力证据转化为简历素材和下一轮画像数据。",
            "actions": [
                "围绕四维能力整理简历素材。",
                "准备优势和短板的自我介绍话术。",
                "重新提交诊断，观察画像变化。"
            ],
            "deliverables": ["能力证据型简历素材", "二次画像对比记录"]
        }
    ]

    learning_tasks = [
        {
            "dimension": item["name"],
            "task": item["action"],
            "timebox": "1周",
            "output": "可检查的学习记录或项目材料"
        }
        for item in development_focus
    ]

    return {
        "summary": (
            f"当前能力画像均值约为 {average_score} 分，"
            f"{ABILITY_DIMENSIONS[strongest_key]['name']}相对突出，"
            f"{ABILITY_DIMENSIONS[weakest_key]['name']}需要优先补强。"
            "本页已按能力画像视角重新组织历史记录，仅保留能力诊断和成长建议。"
        ),
        "advantages": [
            f"{ABILITY_DIMENSIONS[strongest_key]['name']}得分最高，当前为 {ability_scores.get(strongest_key, 0)} 分。",
            (
                f"已识别到显性技能：{'、'.join(recognized_skills[:6])}。"
                if recognized_skills
                else "已保存基础画像信息，可继续补充技能证据。"
            ),
            "能力画像可继续用于成长跟踪和学习任务拆解。"
        ],
        "weaknesses": [
            f"{ABILITY_DIMENSIONS[weakest_key]['name']}当前得分最低，为 {ability_scores.get(weakest_key, 0)} 分。",
            risk_flags[0] if risk_flags else "建议继续补充可量化成果，提升画像可信度。",
            "旧记录缺少完整工作流轨迹，重新诊断后可生成多智能体过程记录。"
        ],
        "profile_tags": profile_tags,
        "risk_flags": risk_flags or ["暂未发现高风险短板，建议继续补充可量化成果。"],
        "evidence_cards": evidence_cards,
        "dimension_insights": dimension_insights,
        "development_focus": development_focus,
        "learning_tasks": learning_tasks,
        "quality_review": [
            "该历史记录已按能力画像视角动态回填。",
            "岗位匹配内容未在画像页展示。",
            "重新诊断后可生成完整多智能体工作流轨迹。"
        ],
        "workflow_steps": [],
        "tool_calls": [],
        "collaboration_log": [],
        "review_findings": [],
        "shared_workspace": {
            "legacy_record": True,
            "note": "旧记录未保存多智能体工具调用链，当前为动态回填画像。"
        },
        "agent_roster": AGENT_ROSTER,
        "llm_agents": [],
        "growth_path": growth_path,
        "used_llm": False,
        "agent_warning": agent_result.get("agent_warning", "")
    }


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

        if not normalized.get("tool_calls"):
            normalized["tool_calls"] = [{
                "tool_name": "LegacyProfileCompatibilityTool",
                "called_by": "画像页兼容适配器",
                "purpose": "兼容旧版或半新版画像记录，标记该记录未保存完整工具调用链",
                "input_summary": "agent_result_json + diagnosis_records scores",
                "output_summary": "已补齐专家角色展示；建议重新诊断生成完整工具链",
                "status": "completed"
            }]

        if not normalized.get("collaboration_log"):
            normalized["collaboration_log"] = [{
                "sender": "画像页兼容适配器",
                "receiver": "前端画像展示",
                "message": "该历史记录没有保存专家交接链，当前仅做兼容展示；重新诊断后会生成完整 LLM 协作记录。",
                "artifact": "legacy_profile_adapter"
            }]

        if not normalized.get("review_findings"):
            normalized["review_findings"] = [{
                "severity": "medium",
                "dimension": "历史记录",
                "finding": "该记录缺少新版工具调用和专家交接证据，建议重新诊断生成完整链路。"
            }]

        return normalized

    return build_ability_profile_fallback(
        student_data=student_data,
        ability_scores=ability_scores,
        agent_result=agent_result
    )


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

    raw_agent_result = load_agent_result(record)

    agent_result = normalize_ability_profile_result(
        student_data=student_data,
        ability_scores=ability_scores,
        agent_result=raw_agent_result
    )

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
            detail = gap_map.get(job.get("job_name"))

            # 岗位名匹配不上时，按 TOP5 顺序兜底
            if detail is None and index < len(top5_gap_paths):
                detail = top5_gap_paths[index]

            job["gap_detail"] = detail or {}

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
            "learning_tasks": agent_result.get("learning_tasks", []),
            "quality_review": agent_result.get("quality_review", []),
            "workflow_steps": agent_result.get("workflow_steps", []),
            "tool_calls": agent_result.get("tool_calls", []),
            "collaboration_log": agent_result.get("collaboration_log", []),
            "review_findings": agent_result.get("review_findings", []),
            "shared_workspace": agent_result.get("shared_workspace", {}),
            "agent_roster": agent_result.get("agent_roster", []),
            "llm_agents": agent_result.get("llm_agents", []),
            "growth_path": agent_result.get("growth_path", []),
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

    result = optimize_resume(
        resume_text=final_resume_text,
        job_description=final_job_description,
        target_role=target_role,
        output_language=output_language,
        harvard_format=harvard_format == "on"
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

    session = build_interview_session(
        target_role=final_target_role,
        job_description=final_job_description,
        resume_text=final_resume_text
    )

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

    result = respond_to_interview_answer(
        session=session,
        question=question,
        answer=answer,
        round_index=round_index,
        history=history
    )

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
            "username": request.session.get("username")
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

    agent_result = run_diagnosis_agent(student_data)
    ai_assessment = score_four_dimensions_llm(student_data)
    if ai_assessment.get("used_llm"):
        agent_result["ability_scores"] = ai_assessment["ability_scores"]
        agent_result["score_evidence"] = ai_assessment["score_evidence"]
        agent_result["recognized_skills"] = ai_assessment.get("recognized_skills", [])
        agent_result["assessment_summary"] = ai_assessment.get("assessment_summary", "")

    ability_scores = agent_result["ability_scores"]

    gap_path_result = generate_top5_gap_paths(
        student_data={
            "name": name,
            "major": major,
            "grade": grade,
            "target_job": target_job,
            "skills": skills,
            "projects": projects,
            "competitions": competitions,
            "certificates": certificates,
            "self_intro": self_intro,
        },
        job_recommendations=agent_result.get("job_recommendations", [])
    )

    agent_result["top5_gap_paths"] = gap_path_result.get("top5_gap_paths", [])
    agent_result["used_llm"] = gap_path_result.get("used_llm", False)
    agent_result["agent_warning"] = gap_path_result.get("agent_warning", "")

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
            gap_path_result = generate_top5_gap_paths(
                student_data=student_data,
                job_recommendations=top5_jobs
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
                detail = gap_map.get(job.get("job_name"))

                # 如果岗位名没匹配上，就按顺序兜底
                if detail is None and index < len(top5_gap_paths):
                    detail = top5_gap_paths[index]

                job["gap_detail"] = detail or {}
            else:
                job["gap_detail"] = {}

    return templates.TemplateResponse(
        request,
        "job_match.html",
        {
            "title": "岗位匹配结果",
            "student": student_data,
            "job_matches": job_matches,
            "username": request.session.get("username")
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
