from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="岗位能力达成学生成长诊断与精准就业智能体系统",
    description="面向学生成长诊断、岗位匹配和个性化路径规划的智能体系统",
    version="0.1.0"
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "岗位能力达成学生成长诊断与精准就业智能体系统"
        }
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "系统运行正常"
    }