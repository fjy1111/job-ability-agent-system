from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app=FastAPI(
    title="岗位能力达成学生成长诊断与精准就业智能体系统",
    description="面向学生成长诊断、岗位匹配和个性化路径规划的智能体系统",
    version="0.1.0"
)
app.mount("/static",StaticFiles(directory="app/static"),name="static")
templates=Jinja2Templates(directory="app/templates")
@app.get("/")
def index(request:Request):
    return templates.TemplateResponse(request,"index.html",{ "title": "岗位能力达成学生成长诊断与精准就业智能体系统"})

@app.get("/student/input")
def student_input(request:Request):
    return templates.TemplateResponse(
        request,
        "student_input.html",
        {
            "title":"学生信息输入"
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
    return templates.TemplateResponse(
        request,
        "student_submit.html",
        {
            "title":"学生信息提交结果",
            "student": student_data,
        }
    )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "系统运行正常"
    }