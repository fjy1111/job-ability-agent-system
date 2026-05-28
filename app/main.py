import json
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="岗位能力达成学生成长诊断与精准就业智能体系统",
    description="面向学生成长诊断、岗位匹配和个性化路径规划的智能体系统",
    version="0.1.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# data 目录
DATA_DIR = BASE_DIR / "data"

# 最新一次提交的学生数据文件
LATEST_STUDENT_FILE = DATA_DIR / "latest_student.json"


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
    self_intro: str = Form("")
):
    """
    接收学生提交的表单数据，并保存为 JSON 文件
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

    # 确保 data 目录存在
    DATA_DIR.mkdir(exist_ok=True)

    # 保存学生数据到 JSON 文件
    with open(LATEST_STUDENT_FILE, "w", encoding="utf-8") as f:
        json.dump(student_data, f, ensure_ascii=False, indent=4)

    # 提交成功后，进入能力画像页面
    return templates.TemplateResponse(
        request,
        "student_submit.html",
        {
            "title": "学生信息提交成功",
            "student": student_data
        }
    )


@app.get("/ability/profile")
def ability_profile(request: Request):
    """
    能力画像页面
    """

    # 如果还没有提交过学生信息，给一个默认空数据
    if not LATEST_STUDENT_FILE.exists():
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
    else:
        with open(LATEST_STUDENT_FILE, "r", encoding="utf-8") as f:
            student_data = json.load(f)

    # 这里先写模拟能力分数，后面会替换成算法模块
    ability_scores = {
        "professional": 75,
        "practice": 70,
        "tools": 80,
        "career": 65
    }

    ability_explain = {
        "professional": "专业基础能力主要来自课程、专业知识和相关证书。",
        "practice": "技术实践能力主要来自项目经历、竞赛经历和实习经历。",
        "tools": "工具技能能力主要来自 Python、Java、Linux、数据库、AI 工具等掌握情况。",
        "career": "职业发展能力主要来自表达能力、简历质量、目标清晰度和面试准备情况。"
    }

    return templates.TemplateResponse(
        request,
        "ability_profile.html",
        {
            "title": "学生能力画像",
            "student": student_data,
            "ability_scores": ability_scores,
            "ability_explain": ability_explain
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