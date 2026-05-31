from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# =========================================================
# 读取环境变量
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


# =========================================================
# 岗位知识库
# 后续可以替换为 MySQL 中的岗位表或真实招聘数据
# =========================================================

JOB_LIBRARY = [
    {
        "job_name": "Python后端开发工程师",
        "category": "软件开发",
        "skills": [
            "Python", "FastAPI", "Django", "Flask",
            "MySQL", "Redis", "Git", "Linux", "Docker"
        ],
        "description": "负责后端接口、数据库设计和业务系统开发。"
    },
    {
        "job_name": "Java后端开发工程师",
        "category": "软件开发",
        "skills": [
            "Java", "Spring Boot", "MySQL",
            "Redis", "Git", "Linux", "Docker"
        ],
        "description": "负责企业级后端服务和业务接口开发。"
    },
    {
        "job_name": "前端开发工程师",
        "category": "软件开发",
        "skills": [
            "HTML", "CSS", "JavaScript", "Vue",
            "React", "Git", "TypeScript"
        ],
        "description": "负责网页界面、交互逻辑与前端工程开发。"
    },
    {
        "job_name": "算法工程师",
        "category": "人工智能",
        "skills": [
            "Python", "PyTorch", "深度学习", "机器学习",
            "算法", "数据结构", "NumPy", "OpenCV"
        ],
        "description": "负责算法设计、训练优化和模型部署。"
    },
    {
        "job_name": "机器学习工程师",
        "category": "人工智能",
        "skills": [
            "Python", "PyTorch", "机器学习", "深度学习",
            "数据分析", "Linux", "Docker"
        ],
        "description": "负责机器学习模型训练、验证和工程部署。"
    },
    {
        "job_name": "数据分析师",
        "category": "数据方向",
        "skills": [
            "Python", "MySQL", "Excel",
            "Pandas", "数据分析", "可视化", "统计学"
        ],
        "description": "负责业务数据处理、分析和可视化汇报。"
    },
    {
        "job_name": "测试开发工程师",
        "category": "软件测试",
        "skills": [
            "Python", "接口测试", "自动化测试",
            "MySQL", "Git", "Linux"
        ],
        "description": "负责自动化测试、接口测试和质量保障平台开发。"
    },
    {
        "job_name": "DevOps运维开发工程师",
        "category": "系统运维",
        "skills": [
            "Linux", "Docker", "Git",
            "Python", "部署", "数据库", "云平台"
        ],
        "description": "负责项目部署、服务监控与自动化运维。"
    },
    {
        "job_name": "产品经理",
        "category": "产品方向",
        "skills": [
            "需求分析", "原型设计", "沟通表达",
            "项目管理", "用户研究"
        ],
        "description": "负责需求分析、产品设计和项目协调。"
    },
    {
        "job_name": "人工智能应用开发工程师",
        "category": "人工智能",
        "skills": [
            "Python", "大模型", "LangChain",
            "LangGraph", "FastAPI", "MySQL", "Git"
        ],
        "description": "负责大模型应用、智能体和业务系统集成开发。"
    }
]


GENERAL_SKILLS = [
    "Python", "Java", "C++", "HTML", "CSS", "JavaScript",
    "Vue", "React", "TypeScript",
    "FastAPI", "Django", "Flask", "Spring Boot",
    "MySQL", "Redis", "数据库", "SQL",
    "Git", "Linux", "Docker", "云平台",
    "PyTorch", "TensorFlow", "OpenCV",
    "机器学习", "深度学习", "算法", "数据结构",
    "数据分析", "Pandas", "NumPy", "可视化", "统计学",
    "大模型", "LangChain", "LangGraph",
    "接口测试", "自动化测试",
    "需求分析", "原型设计", "项目管理", "沟通表达"
]


# =========================================================
# LangGraph 状态定义
# =========================================================

class DiagnosisState(TypedDict, total=False):
    """
    整个智能体执行过程中的共享状态。
    每一个节点都可以读取前面节点的结果，并添加自己的输出。
    """

    student: dict[str, str]

    normalized_text: str
    recognized_skills: list[str]

    ability_scores: dict[str, int]
    score_evidence: dict[str, list[str]]

    job_recommendations: list[dict[str, Any]]

    summary: str
    advantages: list[str]
    weaknesses: list[str]
    job_match_analysis: str
    growth_path: list[dict[str, Any]]

    used_llm: bool
    agent_warning: str


# =========================================================
# 大模型结构化输出格式
# =========================================================

class GrowthStage(BaseModel):
    stage: str = Field(description="阶段名称，例如第一阶段：基础补强")
    duration: str = Field(description="时间范围，例如第1-2个月")
    goal: str = Field(description="本阶段核心目标")
    actions: list[str] = Field(description="本阶段具体行动，建议3项")
    deliverables: list[str] = Field(description="可验证成果，建议1到2项")


class AIReport(BaseModel):
    summary: str = Field(description="针对该学生的整体诊断总结")
    advantages: list[str] = Field(description="学生当前优势，建议2到3项")
    weaknesses: list[str] = Field(description="学生目前不足，建议2到3项")
    job_match_analysis: str = Field(description="结合TOP5岗位结果进行分析")
    growth_path: list[GrowthStage] = Field(description="三阶段成长路径规划")


# =========================================================
# 通用工具函数
# =========================================================

def _safe_text(value: str | None) -> str:
    return (value or "").strip()


def _clamp_score(score: int) -> int:
    return max(0, min(100, score))


def _contains(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()

def extract_json_from_llm_text(text: str) -> dict:
    """
    从大模型返回文本中提取 JSON。
    兼容 ```json ... ``` 包裹的情况。
    """

    text = text.strip()

    # 去掉 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 如果前后有解释文字，截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        return json.loads(json_text)

    raise ValueError("大模型返回内容不是合法 JSON")


def _create_llm() -> ChatOpenAI | None:
    """
    根据 .env 创建大模型。
    支持 DeepSeek 等 OpenAI 兼容接口。
    """

    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not use_llm:
        return None

    if not api_key:
        raise RuntimeError("USE_LLM=true，但没有配置 LLM_API_KEY")

    if not model:
        raise RuntimeError("USE_LLM=true，但没有配置 LLM_MODEL")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.2,
        timeout=60,
        max_retries=2
    )


# =========================================================
# 节点一：提取学生画像
# =========================================================

def extract_profile_node(state: DiagnosisState) -> dict[str, Any]:
    """
    将学生原始表单拼接成可分析文本，并识别明确出现过的技能。
    """

    student = state["student"]

    raw_text = " ".join([
        _safe_text(student.get("major")),
        _safe_text(student.get("target_job")),
        _safe_text(student.get("skills")),
        _safe_text(student.get("projects")),
        _safe_text(student.get("competitions")),
        _safe_text(student.get("certificates")),
        _safe_text(student.get("self_intro"))
    ])

    recognized_skills = [
        skill for skill in GENERAL_SKILLS
        if _contains(raw_text, skill)
    ]

    # 去重，同时保持原顺序
    recognized_skills = list(dict.fromkeys(recognized_skills))

    return {
        "normalized_text": raw_text,
        "recognized_skills": recognized_skills
    }


# =========================================================
# 节点二：能力画像评分
# =========================================================

def score_ability_node(state: DiagnosisState) -> dict[str, Any]:
    """
    使用可解释的规则生成四维能力分数。
    第一版避免让大模型直接随意修改分数，便于展示和答辩。
    """

    student = state["student"]
    text = state["normalized_text"]
    skills = state["recognized_skills"]

    professional_keywords = [
        "数据结构", "算法", "数据库", "计算机网络",
        "操作系统", "机器学习", "深度学习", "统计学"
    ]

    tool_keywords = [
        "Python", "Java", "C++", "FastAPI", "Django",
        "Flask", "Spring Boot", "MySQL", "Redis",
        "Git", "Linux", "Docker", "PyTorch",
        "Vue", "React", "LangGraph"
    ]

    professional_hits = [
        item for item in professional_keywords
        if _contains(text, item)
    ]

    tool_hits = [
        item for item in tool_keywords
        if _contains(text, item)
    ]

    has_project = bool(_safe_text(student.get("projects")))
    has_competition = bool(_safe_text(student.get("competitions")))
    has_certificate = bool(_safe_text(student.get("certificates")))
    has_target_job = bool(_safe_text(student.get("target_job")))
    has_intro = len(_safe_text(student.get("self_intro"))) >= 30

    professional_score = 45 + len(professional_hits) * 6
    if has_certificate:
        professional_score += 8

    practice_score = 35
    if has_project:
        practice_score += 25
    if has_competition:
        practice_score += 18
    if "实习" in text:
        practice_score += 12
    if any(word in text for word in ["开发", "设计", "完成", "实现", "负责"]):
        practice_score += 6

    tools_score = 35 + len(tool_hits) * 5
    if len(skills) >= 6:
        tools_score += 5

    career_score = 35
    if has_target_job:
        career_score += 25
    if has_intro:
        career_score += 12
    if has_certificate:
        career_score += 8
    if has_competition or has_project:
        career_score += 8

    scores = {
        "professional": _clamp_score(professional_score),
        "practice": _clamp_score(practice_score),
        "tools": _clamp_score(tools_score),
        "career": _clamp_score(career_score)
    }

    evidence = {
        "professional": professional_hits or ["暂未识别到明确的专业课程或专业知识证据"],
        "practice": [
            item for item, flag in [
                ("存在项目经历", has_project),
                ("存在竞赛经历", has_competition),
                ("文本中提及实习经历", "实习" in text)
            ]
            if flag
        ] or ["暂未填写项目、竞赛或实习经历"],
        "tools": tool_hits or ["暂未识别到明确工具技能"],
        "career": [
            item for item, flag in [
                ("已填写目标岗位", has_target_job),
                ("具有较完整自我介绍", has_intro),
                ("已填写相关证书", has_certificate)
            ]
            if flag
        ] or ["目标岗位和职业准备信息较少"]
    }

    return {
        "ability_scores": scores,
        "score_evidence": evidence
    }


# =========================================================
# 节点三：岗位匹配和 TOP5 推荐
# =========================================================

def match_jobs_node(state: DiagnosisState) -> dict[str, Any]:
    """
    根据学生技能和目标岗位，从岗位库中选出 TOP5。
    """

    student = state["student"]
    text = state["normalized_text"]
    target_job = _safe_text(student.get("target_job"))

    recommendations: list[dict[str, Any]] = []

    for job in JOB_LIBRARY:
        required_skills = job["skills"]

        matched_skills = [
            skill for skill in required_skills
            if _contains(text, skill)
        ]

        missing_skills = [
            skill for skill in required_skills
            if skill not in matched_skills
        ]

        skill_ratio = len(matched_skills) / len(required_skills)

        target_bonus = 0
        if target_job and (
            target_job in job["job_name"]
            or job["job_name"] in target_job
            or any(word in target_job for word in job["job_name"].replace("工程师", "").split())
        ):
            target_bonus = 15

        project_bonus = 5 if _safe_text(student.get("projects")) else 0

        match_score = round(30 + skill_ratio * 55 + target_bonus + project_bonus)
        match_score = _clamp_score(match_score)

        recommendations.append({
            "job_name": job["job_name"],
            "category": job["category"],
            "match_score": match_score,
            "matched_skills": matched_skills,
            "skill_gaps": missing_skills[:4],
            "description": job["description"]
        })

    recommendations.sort(
        key=lambda item: item["match_score"],
        reverse=True
    )

    return {
        "job_recommendations": recommendations[:5]
    }


# =========================================================
# 默认报告：没有接入大模型时也能正常演示
# =========================================================

def _fallback_report(state: DiagnosisState) -> dict[str, Any]:
    scores = state["ability_scores"]
    top_jobs = state["job_recommendations"]
    student = state["student"]

    score_labels = {
        "professional": "专业基础能力",
        "practice": "技术实践能力",
        "tools": "工具技能能力",
        "career": "职业发展能力"
    }

    strongest_key = max(scores, key=scores.get)
    weakest_key = min(scores, key=scores.get)

    first_job = top_jobs[0]
    gaps = first_job["skill_gaps"]

    target_job = _safe_text(student.get("target_job")) or first_job["job_name"]

    summary = (
        f"根据当前填写的信息，学生在{score_labels[strongest_key]}方面表现相对较好，"
        f"当前优先推荐岗位为“{first_job['job_name']}”，匹配度为"
        f"{first_job['match_score']}分。后续应重点加强"
        f"{score_labels[weakest_key]}，并围绕目标岗位“{target_job}”"
        f"逐步补齐核心技能。"
    )

    advantages = [
        f"{score_labels[strongest_key]}得分较高，当前得分为 {scores[strongest_key]} 分。",
        f"与“{first_job['job_name']}”岗位已有一定基础匹配。",
        "目标岗位与技能信息可以用于继续制定针对性学习计划。"
    ]

    weaknesses = [
        f"{score_labels[weakest_key]}当前得分相对最低，为 {scores[weakest_key]} 分。",
        f"推荐优先补充技能：{'、'.join(gaps) if gaps else '继续强化已有技能并积累项目成果'}。",
        "建议增加可展示的项目成果、竞赛成果或实习经历。"
    ]

    path = [
        {
            "stage": "第一阶段：基础补强",
            "duration": "第1-2个月",
            "goal": f"围绕 {target_job} 补足岗位基础能力",
            "actions": [
                f"系统学习岗位缺失技能：{'、'.join(gaps[:2]) if gaps else '专业基础知识'}",
                "整理已有技能清单和学习笔记",
                "完成至少一个基础练习项目"
            ],
            "deliverables": [
                "技能学习笔记",
                "基础项目代码仓库"
            ]
        },
        {
            "stage": "第二阶段：项目实践",
            "duration": "第3-4个月",
            "goal": "形成可展示的岗位相关项目成果",
            "actions": [
                f"围绕 {first_job['job_name']} 设计一个综合项目",
                "使用 Git 管理开发过程并完善 README",
                "总结项目中使用的技术与解决的问题"
            ],
            "deliverables": [
                "完整项目作品",
                "项目介绍文档"
            ]
        },
        {
            "stage": "第三阶段：就业准备",
            "duration": "第5-6个月",
            "goal": "完成简历优化和岗位投递准备",
            "actions": [
                "根据推荐岗位修改简历中的技能和项目描述",
                "整理常见面试题并进行模拟面试",
                "选择匹配度较高的岗位进行投递"
            ],
            "deliverables": [
                "岗位定制简历",
                "面试准备清单"
            ]
        }
    ]

    return {
        "summary": summary,
        "advantages": advantages,
        "weaknesses": weaknesses,
        "job_match_analysis": (
            f"当前 TOP1 推荐岗位为“{first_job['job_name']}”，"
            f"系统识别到的已匹配技能包括："
            f"{'、'.join(first_job['matched_skills']) if first_job['matched_skills'] else '暂无明确技能证据'}。"
            f"仍需补充：{'、'.join(gaps) if gaps else '暂无明显缺口'}。"
        ),
        "growth_path": path,
        "used_llm": False
    }


# =========================================================
# 节点四：大模型生成诊断总结和路径规划
# =========================================================

def generate_report_node(state: DiagnosisState) -> dict[str, Any]:
    """
    使用大模型生成诊断总结和成长路径。
    DeepSeek 普通聊天调用可用，但 with_structured_output 可能报 BadRequestError，
    所以这里改成普通 JSON 文本输出，再由 Python 解析。
    """

    llm = _create_llm()

    if llm is None:
        return _fallback_report(state)

    student = state["student"]
    scores = state["ability_scores"]
    recommendations = state["job_recommendations"]
    recognized_skills = state["recognized_skills"]

    prompt = f"""
你是“岗位能力达成学生成长诊断与精准就业智能体系统”的职业诊断专家。

请根据学生信息、四维能力分数和TOP5岗位推荐结果，生成诊断报告。

重要要求：
1. 不得编造学生没有填写的证书、竞赛、实习或项目。
2. 四维能力分数已经由系统算法计算完成，不允许修改。
3. TOP5 岗位排序和匹配分数已经由系统计算完成，不允许修改。
4. 你的任务是解释结果，并生成具体、可执行的成长路径。
5. 必须只输出 JSON，不要输出 Markdown，不要输出解释文字，不要使用 ```json 代码块。
6. growth_path 必须包含三个阶段。
7. 每个阶段必须包含 stage、duration、goal、actions、deliverables 五个字段。
8. actions 建议 3 条，deliverables 建议 2 条。

学生信息：
{json.dumps(student, ensure_ascii=False)}

系统识别技能：
{json.dumps(recognized_skills, ensure_ascii=False)}

四维能力分数：
{json.dumps(scores, ensure_ascii=False)}

TOP5岗位推荐：
{json.dumps(recommendations, ensure_ascii=False)}

请严格按照下面 JSON 格式输出：

{{
  "summary": "整体诊断总结，100到200字",
  "advantages": [
    "优势1",
    "优势2",
    "优势3"
  ],
  "weaknesses": [
    "短板1",
    "短板2",
    "短板3"
  ],
  "job_match_analysis": "结合TOP5岗位推荐结果进行分析，说明为什么推荐这些岗位以及主要差距",
  "growth_path": [
    {{
      "stage": "第一阶段：基础补强",
      "duration": "第1-2个月",
      "goal": "本阶段目标",
      "actions": [
        "行动任务1",
        "行动任务2",
        "行动任务3"
      ],
      "deliverables": [
        "验收成果1",
        "验收成果2"
      ]
    }},
    {{
      "stage": "第二阶段：项目实践",
      "duration": "第3-4个月",
      "goal": "本阶段目标",
      "actions": [
        "行动任务1",
        "行动任务2",
        "行动任务3"
      ],
      "deliverables": [
        "验收成果1",
        "验收成果2"
      ]
    }},
    {{
      "stage": "第三阶段：就业准备",
      "duration": "第5-6个月",
      "goal": "本阶段目标",
      "actions": [
        "行动任务1",
        "行动任务2",
        "行动任务3"
      ],
      "deliverables": [
        "验收成果1",
        "验收成果2"
      ]
    }}
  ]
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content

        report = extract_json_from_llm_text(content)

        return {
            "summary": report.get("summary", ""),
            "advantages": report.get("advantages", []),
            "weaknesses": report.get("weaknesses", []),
            "job_match_analysis": report.get("job_match_analysis", ""),
            "growth_path": report.get("growth_path", []),
            "used_llm": True,
            "agent_warning": ""
        }

    except Exception as exc:
        fallback = _fallback_report(state)
        fallback["agent_warning"] = (
            f"大模型调用失败，当前展示规则版诊断结果：{type(exc).__name__}: {exc}"
        )
        return fallback

# =========================================================
# 构建 LangGraph 智能体
# =========================================================

def build_diagnosis_graph():
    builder = StateGraph(DiagnosisState)

    builder.add_node("extract_profile", extract_profile_node)
    builder.add_node("score_ability", score_ability_node)
    builder.add_node("match_jobs", match_jobs_node)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "score_ability")
    builder.add_edge("score_ability", "match_jobs")
    builder.add_edge("match_jobs", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


diagnosis_graph = build_diagnosis_graph()


# =========================================================
# 对外调用函数
# =========================================================

def run_diagnosis_agent(student_data: dict[str, str]) -> dict[str, Any]:
    """
    main.py 只需要调用这个函数即可获得完整诊断结果。
    """

    result = diagnosis_graph.invoke({
        "student": student_data
    })

    return {
        "ability_scores": result["ability_scores"],
        "score_evidence": result["score_evidence"],
        "recognized_skills": result["recognized_skills"],
        "job_recommendations": result["job_recommendations"],
        "summary": result["summary"],
        "advantages": result["advantages"],
        "weaknesses": result["weaknesses"],
        "job_match_analysis": result["job_match_analysis"],
        "growth_path": result["growth_path"],
        "used_llm": result.get("used_llm", False),
        "agent_warning": result.get("agent_warning", "")
    }