from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

from app.services.llm_errors import LLMCallError


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


DEFAULT_INTERVIEWER = "林老师"
QUESTION_COUNT = 6
WRITTEN_QUESTION_POOL_SIZE = 6
WRITTEN_POOL_CACHE_TTL_SECONDS = 60 * 60
WRITTEN_POOL_CACHE_MAX_ITEMS = 64
_WRITTEN_POOL_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

ROLE_KEYWORDS = [
    "Python", "Java", "Spring Boot", "FastAPI", "Django", "MySQL", "Redis",
    "Linux", "Docker", "Git", "Vue", "React", "TypeScript", "JavaScript",
    "数据分析", "机器学习", "深度学习", "算法", "大模型", "LangChain",
    "LangGraph", "项目管理", "需求分析", "沟通表达", "自动化测试", "接口测试",
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        raise LLMCallError()

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise LLMCallError()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise LLMCallError()

    if not isinstance(data, dict):
        raise LLMCallError()
    return data


def _create_llm() -> Any:
    if ChatOpenAI is None:
        raise LLMCallError()
    if os.getenv("USE_LLM", "true").lower() != "true":
        raise LLMCallError()

    api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("LLM_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or os.getenv("DASHSCOPE_BASE_URL", "").strip()
    )
    model = (
        os.getenv("INTERVIEW_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("DASHSCOPE_MODEL", "").strip()
    )

    if not api_key or not model:
        raise LLMCallError()

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.35,
        timeout=60,
        max_retries=0,
    )


def _pick_focus_keywords(text: str, limit: int = 8) -> list[str]:
    focus = [keyword for keyword in ROLE_KEYWORDS if keyword.lower() in text.lower()]
    raw_items = re.split(r"[\s,，、;；。:：/|]+", text)
    for item in raw_items:
        word = item.strip("()（）[]【】<>《》")
        if len(word) < 2 or len(word) > 20:
            continue
        if word not in focus and re.search(r"[\u4e00-\u9fffA-Za-z]", word):
            focus.append(word)
        if len(focus) >= limit:
            break
    return focus[:limit]


def _infer_role(target_role: str, job_description: str, resume_text: str) -> str:
    target_role = _safe_text(target_role)
    if target_role:
        return target_role

    combined = f"{job_description} {resume_text}"
    role_patterns = [
        r"([A-Za-z0-9\u4e00-\u9fff]{2,24}(?:工程师|开发|分析师|产品经理|测试|运营|设计师))",
        r"(?:岗位|职位|目标)[:：\s]*([A-Za-z0-9\u4e00-\u9fff]{2,24})",
    ]
    for pattern in role_patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(1)
    return "目标岗位"


def _generate_llm_questions(role: str, job_description: str, resume_text: str) -> list[str]:
    prompt = f"""
你是严谨但友好的 AI 模拟面试官。请根据目标岗位、JD 和候选人简历，生成 {QUESTION_COUNT} 个循序渐进的中文面试问题。

要求：
1. 问题要像真人面试官一样自然、具体。
2. 覆盖自我介绍、项目经历、岗位技能、问题拆解、短板反思、录用理由。
3. 不要编造候选人没有提供的信息。
4. 只输出 JSON，不要 Markdown。

目标岗位：{role}
岗位信息：{job_description[:2000]}
简历信息：{resume_text[:3000]}

JSON 格式：
{{
  "questions": ["问题1", "问题2", "问题3", "问题4", "问题5", "问题6"]
}}
"""
    try:
        response = _create_llm().invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        questions = [
            _safe_text(item)
            for item in parsed.get("questions", [])
            if _safe_text(item)
        ]
        if len(questions) < QUESTION_COUNT:
            raise LLMCallError()
        return questions[:QUESTION_COUNT]
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def _normalize_written_question(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    prompt = _safe_text(item.get("question"))
    options = item.get("options")
    if not prompt or not isinstance(options, list):
        return None

    normalized_options = [_safe_text(option) for option in options]
    if len(normalized_options) != 4 or any(not option for option in normalized_options):
        return None
    if len(set(normalized_options)) != 4:
        return None

    correct_index = item.get("correct_index")
    if isinstance(correct_index, str):
        value = correct_index.strip().upper()
        if value in {"A", "B", "C", "D"}:
            correct_index = ord(value) - ord("A")
        else:
            try:
                correct_index = int(value)
            except ValueError:
                return None

    if not isinstance(correct_index, int) or not 0 <= correct_index < 4:
        return None

    return {
        "question": prompt,
        "options": normalized_options,
        "correct_index": correct_index,
        "explanation": _safe_text(item.get("explanation")) or "请结合岗位知识点复习该题。",
        "category": _safe_text(item.get("category")) or "岗位基础",
    }


def _generate_llm_written_question_pool(
    role: str,
    job_description: str,
    resume_text: str,
) -> list[dict[str, Any]]:
    prompt = f"""
你是一名企业招聘笔试命题专家。请根据目标岗位、岗位描述和候选人简历，生成 {WRITTEN_QUESTION_POOL_SIZE} 道中文单选题，供系统随机抽取。
要求：
1. 每题只有一个正确答案，固定提供 4 个互不重复的选项。
2. 题目覆盖岗位基础、项目实践、工程规范和问题分析，不要考察简历中完全无关的知识。
3. 难度由基础到进阶，避免歧义、脑筋急转弯和纯记忆冷知识。
4. correct_index 必须是 0、1、2、3 之一，分别对应第 1 至第 4 个选项。
5. 只输出 JSON，不要 Markdown。

目标岗位：{role}
岗位信息：{job_description[:2200]}
简历信息：{resume_text[:3200]}

JSON 格式：
{{
  "questions": [
    {{
      "question": "题目",
      "options": ["选项A", "选项B", "选项C", "选项D"],
      "correct_index": 0,
      "explanation": "答案解析",
      "category": "知识分类"
    }}
  ]
}}
"""
    try:
        response = _create_llm().invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        raw_questions = parsed.get("questions")
        if not isinstance(raw_questions, list):
            raise LLMCallError()

        questions = [
            question
            for question in (_normalize_written_question(item) for item in raw_questions)
            if question is not None
        ]
        if len(questions) < QUESTION_COUNT:
            raise LLMCallError()
        return questions
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def _build_local_written_question_pool(
    role: str,
    job_description: str,
    resume_text: str,
) -> list[dict[str, Any]]:
    """按岗位与简历关键词从本地高质量题库即时组卷。"""
    context = f"{role} {job_description} {resume_text}".lower()
    bank: list[tuple[list[str], dict[str, Any]]] = [
        (["java"], {
            "question": "在 Java 中，以下哪个集合更适合高并发读写场景？",
            "options": ["HashMap", "ConcurrentHashMap", "ArrayList", "LinkedList"],
            "correct_index": 1,
            "explanation": "ConcurrentHashMap 针对并发访问进行了设计，适合多线程读写。",
            "category": "Java 基础",
        }),
        (["spring boot", "spring"], {
            "question": "Spring Boot 中存在多个同类型 Bean 时，通常用哪个注解指定注入对象？",
            "options": ["@Order", "@Qualifier", "@Scope", "@DependsOn"],
            "correct_index": 1,
            "explanation": "@Qualifier 可与依赖注入注解配合，按名称明确选择 Bean。",
            "category": "Spring Boot",
        }),
        (["mysql", "sql", "数据库"], {
            "question": "分析 MySQL 查询是否使用索引，最常用的命令是？",
            "options": ["EXPLAIN", "SHOW TABLES", "DESCRIBE", "SHOW CREATE TABLE"],
            "correct_index": 0,
            "explanation": "EXPLAIN 会展示执行计划、索引使用和预估扫描行数。",
            "category": "数据库",
        }),
        (["redis", "缓存"], {
            "question": "为降低缓存击穿风险，热点数据失效时更合适的做法是？",
            "options": ["永久不过期", "互斥重建并设置合理过期时间", "删除数据库索引", "关闭缓存"],
            "correct_index": 1,
            "explanation": "互斥重建可避免大量请求同时回源，并应配合合理的过期策略。",
            "category": "缓存设计",
        }),
        (["docker", "容器"], {
            "question": "将本地 Docker 镜像推送到远程仓库使用哪个命令？",
            "options": ["docker pull", "docker save", "docker push", "docker commit"],
            "correct_index": 2,
            "explanation": "docker push 用于把已正确标记的本地镜像推送到远程仓库。",
            "category": "Docker",
        }),
        (["python"], {
            "question": "Python 中管理项目依赖并隔离运行环境，推荐使用什么？",
            "options": ["虚拟环境", "全局变量", "系统临时目录", "线程锁"],
            "correct_index": 0,
            "explanation": "虚拟环境可以隔离不同项目的解释器依赖与版本。",
            "category": "Python 基础",
        }),
        (["fastapi"], {
            "question": "FastAPI 中声明请求参数和数据校验通常依赖哪个组件？",
            "options": ["Pydantic 模型", "Jinja2 模板", "SQLite 游标", "CSS 选择器"],
            "correct_index": 0,
            "explanation": "FastAPI 使用 Pydantic 模型完成结构声明、解析和数据校验。",
            "category": "FastAPI",
        }),
        (["vue", "react", "javascript", "typescript", "前端"], {
            "question": "前端列表渲染时为元素设置稳定 key，主要作用是什么？",
            "options": ["加密请求", "帮助框架准确复用和更新节点", "压缩图片", "创建数据库索引"],
            "correct_index": 1,
            "explanation": "稳定 key 帮助虚拟 DOM 识别节点身份，减少错误复用和不必要更新。",
            "category": "前端工程",
        }),
        (["机器学习", "深度学习", "算法", "大模型"], {
            "question": "评估分类模型时，类别极不均衡的情况下更应关注哪个指标组合？",
            "options": ["仅准确率", "精确率、召回率与 F1", "仅训练时长", "参数文件大小"],
            "correct_index": 1,
            "explanation": "类别不均衡时准确率可能具有误导性，应结合精确率、召回率和 F1。",
            "category": "机器学习",
        }),
        (["测试", "自动化"], {
            "question": "接口自动化测试中，为保证用例可重复执行，最重要的实践之一是？",
            "options": ["依赖固定脏数据", "准备和清理独立测试数据", "跳过断言", "只测试成功路径"],
            "correct_index": 1,
            "explanation": "独立准备并清理测试数据可减少用例之间的耦合和偶发失败。",
            "category": "测试工程",
        }),
    ]
    general_questions = [
        {
            "question": "以下哪种 HTTP 方法最符合删除 REST 资源的语义？",
            "options": ["GET", "POST", "DELETE", "PATCH"],
            "correct_index": 2,
            "explanation": "DELETE 用于表达删除指定资源的操作。",
            "category": "接口设计",
        },
        {
            "question": "线上接口突然变慢时，合理的第一步是什么？",
            "options": ["直接重写系统", "先查看监控、日志并定位瓶颈", "删除数据库", "忽略告警"],
            "correct_index": 1,
            "explanation": "先依据监控、链路与日志缩小问题范围，再采取针对性措施。",
            "category": "故障排查",
        },
        {
            "question": "团队协作开发中，提交代码前更推荐的做法是？",
            "options": ["跳过测试直接合并", "运行测试并进行代码审查", "删除提交记录", "共享个人密码"],
            "correct_index": 1,
            "explanation": "自动化测试和代码审查可以降低缺陷进入主分支的概率。",
            "category": "工程规范",
        },
        {
            "question": "防止 SQL 注入最有效的基础措施是？",
            "options": ["拼接用户输入", "使用参数化查询", "隐藏按钮", "增加页面颜色"],
            "correct_index": 1,
            "explanation": "参数化查询把 SQL 结构与输入数据分离，是防止 SQL 注入的基础措施。",
            "category": "安全基础",
        },
        {
            "question": "项目成果在简历或面试中怎样表达更有说服力？",
            "options": ["只说参与过", "说明行动、技术方案和量化结果", "省略个人职责", "只罗列工具名"],
            "correct_index": 1,
            "explanation": "清晰说明个人行动、方案与可验证结果更能体现实际能力。",
            "category": "项目实践",
        },
        {
            "question": "设计高并发接口时，哪项做法通常有助于提高稳定性？",
            "options": ["无限制重试", "限流、超时和降级", "取消日志", "关闭监控"],
            "correct_index": 1,
            "explanation": "限流、超时与降级能避免局部故障扩散并保护核心资源。",
            "category": "系统设计",
        },
    ]

    selected: list[dict[str, Any]] = []
    for keywords, question in bank:
        if any(keyword.lower() in context for keyword in keywords):
            selected.append(copy.deepcopy(question))
    selected.extend(copy.deepcopy(general_questions))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in selected:
        if question["question"] in seen:
            continue
        seen.add(question["question"])
        unique.append(question)
    return unique


def _written_pool_cache_key(role: str, job_description: str, resume_text: str) -> str:
    payload = json.dumps(
        {
            "role": role.strip().lower(),
            "job_description": job_description.strip(),
            "resume_text": resume_text.strip(),
            "question_count": WRITTEN_QUESTION_POOL_SIZE,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_or_generate_written_question_pool(
    role: str,
    job_description: str,
    resume_text: str,
) -> list[dict[str, Any]]:
    """相同简历与岗位复用题池，展示时仍重新打乱题目和选项。"""
    now = time.monotonic()
    stale_keys = [
        key
        for key, (created_at, _questions) in _WRITTEN_POOL_CACHE.items()
        if now - created_at > WRITTEN_POOL_CACHE_TTL_SECONDS
    ]
    for key in stale_keys:
        _WRITTEN_POOL_CACHE.pop(key, None)

    cache_key = _written_pool_cache_key(role, job_description, resume_text)
    cached = _WRITTEN_POOL_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached[1])

    use_llm = os.getenv("INTERVIEW_WRITTEN_USE_LLM", "false").lower() == "true"
    if use_llm:
        try:
            questions = _generate_llm_written_question_pool(
                role=role,
                job_description=job_description,
                resume_text=resume_text,
            )
        except LLMCallError:
            questions = _build_local_written_question_pool(role, job_description, resume_text)
    else:
        questions = _build_local_written_question_pool(role, job_description, resume_text)
    if len(_WRITTEN_POOL_CACHE) >= WRITTEN_POOL_CACHE_MAX_ITEMS:
        oldest_key = min(_WRITTEN_POOL_CACHE, key=lambda key: _WRITTEN_POOL_CACHE[key][0])
        _WRITTEN_POOL_CACHE.pop(oldest_key, None)
    _WRITTEN_POOL_CACHE[cache_key] = (now, copy.deepcopy(questions))
    return questions


def build_written_exam(
    target_role: str,
    job_description: str,
    resume_text: str,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> dict[str, Any]:
    """生成一份与岗位和简历相关、题序与选项顺序随机的六题笔试。"""
    role = _infer_role(target_role, job_description, resume_text)
    question_pool = _load_or_generate_written_question_pool(role, job_description, resume_text)
    randomizer = rng or random.SystemRandom()
    selected = (
        randomizer.sample(question_pool, QUESTION_COUNT)
        if len(question_pool) > QUESTION_COUNT
        else list(question_pool)
    )
    randomizer.shuffle(selected)

    questions: list[dict[str, Any]] = []
    for index, source in enumerate(selected, start=1):
        option_pairs = list(enumerate(source["options"]))
        randomizer.shuffle(option_pairs)
        correct_index = next(
            position
            for position, (original_index, _option) in enumerate(option_pairs)
            if original_index == source["correct_index"]
        )
        questions.append({
            "id": f"q{index}",
            "question": source["question"],
            "options": [option for _original_index, option in option_pairs],
            "correct_index": correct_index,
            "explanation": source["explanation"],
            "category": source["category"],
        })

    return {
        "exam_id": str(uuid.uuid4()),
        "target_role": role,
        "job_description": job_description[:2200],
        "resume_text": resume_text[:3200],
        "questions": questions,
        "total_questions": len(questions),
    }


def grade_written_exam(
    exam_session: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    """批改笔试并返回逐题结果，空题按错误处理。"""
    questions = exam_session.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise ValueError("笔试会话中没有题目")
    if not isinstance(answers, dict):
        answers = {}

    correct_count = 0
    details: list[dict[str, Any]] = []
    for question in questions:
        question_id = _safe_text(question.get("id"))
        try:
            selected_index = int(answers.get(question_id))
        except (TypeError, ValueError):
            selected_index = -1
        correct_index = int(question.get("correct_index", -1))
        is_correct = selected_index == correct_index
        correct_count += int(is_correct)
        options = question.get("options") or []
        details.append({
            "id": question_id,
            "question": _safe_text(question.get("question")),
            "category": _safe_text(question.get("category")),
            "selected_index": selected_index,
            "correct_index": correct_index,
            "selected_answer": options[selected_index] if 0 <= selected_index < len(options) else "未作答",
            "correct_answer": options[correct_index] if 0 <= correct_index < len(options) else "",
            "is_correct": is_correct,
            "explanation": _safe_text(question.get("explanation")),
        })

    total = len(questions)
    return {
        "correct_count": correct_count,
        "total_questions": total,
        "score": round(correct_count / total * 100),
        "details": details,
    }


def build_interview_session(
    target_role: str,
    job_description: str,
    resume_text: str,
) -> dict[str, Any]:
    role = _infer_role(target_role, job_description, resume_text)
    combined_context = f"{role} {job_description} {resume_text}"
    focus_keywords = _pick_focus_keywords(combined_context)
    questions = _generate_llm_questions(
        role=role,
        job_description=job_description,
        resume_text=resume_text,
    )

    return {
        "session_id": str(uuid.uuid4()),
        "interviewer_name": DEFAULT_INTERVIEWER,
        "target_role": role,
        "job_description": job_description[:2000],
        "resume_text": resume_text[:3000],
        "focus_keywords": focus_keywords,
        "questions": questions,
        "total_rounds": len(questions),
        "current_round": 1,
        "current_question": questions[0],
        "opening_message": (
            f"你好，我是{DEFAULT_INTERVIEWER}。今天我们围绕“{role}”做一次模拟面试。"
            "我会像正式面试一样提问，也会在每次回答后给你具体建议。"
        ),
    }


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _score_answer(answer: str, focus_keywords: list[str]) -> dict[str, int]:
    length = len(answer)
    has_context = _has_any(answer, ["背景", "需求", "目标", "任务", "问题", "场景"])
    has_action = _has_any(answer, ["负责", "完成", "设计", "实现", "推进", "协作", "使用", "优化", "分析"])
    has_result = bool(re.search(r"\d|%|提升|降低|上线|交付|通过|获奖|用户|准确率|效率|排名|完成", answer))
    matched_focus = [keyword for keyword in focus_keywords if keyword and keyword.lower() in answer.lower()]

    structure_score = 45 + (15 if has_context else 0) + (20 if has_action else 0) + (20 if has_result else 0)
    relevance_score = 58 + min(len(matched_focus), 4) * 8
    relevance_score += 8 if length >= 80 else 0
    relevance_score -= 18 if length < 35 else 0
    evidence_score = 42 + (25 if has_action else 0) + (25 if has_result else 0) + (8 if re.search(r"\d", answer) else 0)
    expression_score = 55 + (22 if 80 <= length <= 450 else 10 if length > 450 else 0)
    expression_score += 8 if "。" in answer or "，" in answer else 0

    scores = {
        "structure_score": _clamp(structure_score),
        "relevance_score": _clamp(relevance_score),
        "evidence_score": _clamp(evidence_score),
        "expression_score": _clamp(expression_score),
    }
    scores["overall_score"] = round(sum(scores.values()) / len(scores))
    return scores


def _generate_llm_feedback(
    session: dict[str, Any],
    question: str,
    answer: str,
    scores: dict[str, int],
) -> dict[str, Any]:
    prompt = f"""
你是 AI 模拟面试官。请根据当前问题和候选人回答，给出简洁、具体、可执行的中文面试反馈。

要求：
1. 像真人面试官一样自然。
2. 不要虚构候选人没有说过的经历。
3. 必须指出回答优点和下一步改进。
4. 只输出 JSON，不要 Markdown。

目标岗位：{session.get("target_role", "目标岗位")}
岗位关键词：{json.dumps(session.get("focus_keywords", []), ensure_ascii=False)}
当前问题：{question}
候选人回答：{answer}
结构化评分：{json.dumps(scores, ensure_ascii=False)}

JSON 格式：
{{
  "feedback_text": "一句总体点评",
  "strengths": ["优点1", "优点2"],
  "suggestions": ["建议1", "建议2", "建议3"],
  "polished_answer": "一段可参考的补强表达"
}}
"""
    try:
        response = _create_llm().invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        required = ["feedback_text", "strengths", "suggestions", "polished_answer"]
        if any(key not in parsed for key in required):
            raise LLMCallError()
        return {
            **scores,
            "feedback_text": _safe_text(parsed.get("feedback_text")),
            "strengths": parsed.get("strengths") if isinstance(parsed.get("strengths"), list) else [],
            "suggestions": parsed.get("suggestions") if isinstance(parsed.get("suggestions"), list) else [],
            "polished_answer": _safe_text(parsed.get("polished_answer")),
            "used_llm": True,
            "question": question,
        }
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def _build_final_report(completed_items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        item.get("feedback", {}).get("overall_score")
        for item in completed_items
        if isinstance(item.get("feedback"), dict)
    ]
    valid_scores = [int(score) for score in scores if isinstance(score, int)]
    average_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

    all_suggestions: list[str] = []
    for item in completed_items:
        feedback = item.get("feedback", {})
        if isinstance(feedback, dict):
            all_suggestions.extend(feedback.get("suggestions", []))

    return {
        "average_score": average_score,
        "summary": "本次模拟面试已完成，建议把高频问题、项目证据和量化结果继续沉淀成固定表达。",
        "improvements": list(dict.fromkeys(all_suggestions))[:4],
        "next_actions": [
            "整理 2 个可反复讲的项目故事，每个故事写清背景、行动、结果。",
            "为目标岗位准备 8-10 个高频问题，并按 STAR 结构复盘。",
            "把回答中的结果补成可验证指标，如数据、交付物、排名或反馈。",
        ],
    }


def respond_to_interview_answer(
    session: dict[str, Any],
    question: str,
    answer: str,
    round_index: int,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    answer = _safe_text(answer)
    history = history or []
    questions = session.get("questions") or []
    focus_keywords = session.get("focus_keywords", [])
    if not isinstance(focus_keywords, list):
        focus_keywords = []

    feedback = _generate_llm_feedback(
        session=session,
        question=question,
        answer=answer,
        scores=_score_answer(answer, focus_keywords),
    )

    completed_item = {
        "round_index": round_index,
        "question": question,
        "answer": answer,
        "feedback": feedback,
    }
    completed_items = [*history, completed_item]

    next_round = round_index + 1
    total_rounds = len(questions)
    finished = next_round > total_rounds
    next_question = "" if finished else questions[next_round - 1]

    session_update = dict(session)
    session_update["current_round"] = next_round if not finished else total_rounds
    session_update["current_question"] = next_question

    return {
        "feedback": feedback,
        "next_question": next_question,
        "next_round": next_round,
        "total_rounds": total_rounds,
        "finished": finished,
        "final_report": _build_final_report(completed_items) if finished else None,
        "session": session_update,
        "history_item": completed_item,
    }
