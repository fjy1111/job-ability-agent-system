from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.services.resume_expert_kb_service import retrieve_resume_expert_rules
from app.services.llm_errors import LLMCallError

load_dotenv()


TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}
MAX_PDF_PAGES = 5


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _first_pdf_pages(pages):
    for index, page in enumerate(pages):
        if index >= MAX_PDF_PAGES:
            break
        yield page


def _extract_docx_text(file_content: bytes) -> str:
    try:
        from docx import Document
    except Exception:
        return ""

    document = Document(BytesIO(file_content))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    table_cells: list[str] = []
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text:
                    table_cells.append(text)
    return "\n".join([*paragraphs, *table_cells]).strip()


def _extract_pdf_text(file_content: bytes) -> tuple[str, str]:
    """
    多解析器兜底提取 PDF 文本。
    优先 pypdf，其次 PyPDF2，再其次 pdfplumber；只读取前几页以避免上传后长时间等待。
    """
    missing_parsers: list[str] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_content))
        text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(reader.pages))
        if text.strip():
            return text.strip(), ""
    except ModuleNotFoundError:
        missing_parsers.append("pypdf")
    except Exception as exc:
        return "", f"PDF 解析失败：{type(exc).__name__}，请粘贴简历文本。"

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(file_content))
        text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(reader.pages))
        if text.strip():
            return text.strip(), ""
    except ModuleNotFoundError:
        missing_parsers.append("PyPDF2")
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(BytesIO(file_content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(pdf.pages))
        if text.strip():
            return text.strip(), ""
    except ModuleNotFoundError:
        missing_parsers.append("pdfplumber")
    except Exception:
        pass

    if missing_parsers:
        return "", "PDF 解析依赖未安装，请在 agent 环境执行：python -m pip install pypdf，然后重启项目。"

    return "", "PDF 未提取到有效文本，可能是扫描件或图片版简历，请粘贴简历文本。"


def extract_resume_text_from_upload(filename: str, file_content: bytes) -> tuple[str, list[str]]:
    """
    从上传文件中提取简历文本。只负责文件解析，不生成替代优化结果。
    """
    suffix = Path(filename).suffix.lower()
    warnings: list[str] = []

    if suffix in TEXT_EXTENSIONS:
        return _decode_text(file_content).strip(), warnings

    if suffix in DOCX_EXTENSIONS:
        text = _extract_docx_text(file_content)
        if text:
            return text, warnings
        warnings.append("Word 文件暂时无法解析，请粘贴简历文本。")
        return "", warnings

    if suffix in PDF_EXTENSIONS:
        text, parser_warning = _extract_pdf_text(file_content)
        if text:
            return text, warnings
        warnings.append(parser_warning)
        return "", warnings

    warnings.append("该文件暂时无法解析，请粘贴简历文本。")
    return "", warnings


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


def _create_llm() -> ChatOpenAI:
    if os.getenv("USE_LLM", "true").lower() != "true":
        raise LLMCallError()

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    model = (
        os.getenv("RESUME_OPTIMIZER_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or os.getenv("DASHSCOPE_MODEL")
    )
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )

    if not api_key or not model:
        raise LLMCallError()

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.25,
        "timeout": 60,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _score(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        raise LLMCallError()
    return max(0, min(100, number))


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _validate_result(data: dict[str, Any]) -> dict[str, Any]:
    required = [
        "overall_score",
        "keyword_score",
        "structure_score",
        "summary",
        "optimized_resume",
        "strengths",
        "weaknesses",
        "rewrite_suggestions",
        "action_items",
    ]
    if any(key not in data for key in required):
        raise LLMCallError()

    rewrite_suggestions = _list(data.get("rewrite_suggestions"))
    if not all(isinstance(item, dict) for item in rewrite_suggestions):
        raise LLMCallError()

    return {
        "overall_score": _score(data.get("overall_score")),
        "keyword_score": _score(data.get("keyword_score")),
        "structure_score": _score(data.get("structure_score")),
        "summary": str(data.get("summary", "")).strip(),
        "optimized_resume": str(data.get("optimized_resume", "")).strip(),
        "strengths": [str(item).strip() for item in _list(data.get("strengths")) if str(item).strip()],
        "weaknesses": [str(item).strip() for item in _list(data.get("weaknesses")) if str(item).strip()],
        "rewrite_suggestions": rewrite_suggestions,
        "action_items": [str(item).strip() for item in _list(data.get("action_items")) if str(item).strip()],
        "agent_warning": "",
    }


def _public_expert_rule(rule: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(rule.get("id", "")).strip(),
        "title": str(rule.get("title", "")).strip(),
        "category": str(rule.get("category", "")).strip(),
        "suggestion": str(rule.get("suggestion", "")).strip(),
        "source_image": str(rule.get("source_image", "")).strip(),
    }


def _expert_rules_prompt(matched_rules: list[dict]) -> str:
    if not matched_rules:
        return ""

    rule_lines = []
    for index, rule in enumerate(matched_rules, start=1):
        title = str(rule.get("title", "")).strip()
        suggestion = str(rule.get("suggestion", "")).strip()
        if title and suggestion:
            rule_lines.append(f"{index}. {title}：{suggestion}")
        elif suggestion:
            rule_lines.append(f"{index}. {suggestion}")

    if not rule_lines:
        return ""

    return f"""
【专家手工标注建议知识库】
以下是从历史专家批注中检索到的规则，请优先参考：

{chr(10).join(rule_lines)}

生成简历优化建议时，请结合学生简历内容和上述专家规则，重点检查：
* 是否缺少岗位名称
* 是否缺少项目时间
* 是否缺少个人角色
* 项目功能是否描述清楚
* 技术亮点是否写清楚“技术方案 + 解决问题 + 效果”
* 是否有量化成果
* 技能描述是否过泛
* 教育经历和证书表达是否规范
"""


def _attach_expert_rules(result: dict[str, Any], matched_rules: list[dict]) -> dict[str, Any]:
    public_rules = [
        rule
        for rule in (_public_expert_rule(item) for item in matched_rules)
        if rule["title"] or rule["suggestion"]
    ]
    result["expert_rules_used_count"] = len(public_rules)
    result["expert_rules_used"] = public_rules
    return result


def _invoke_optimizer_with_retry(prompt: str) -> dict[str, Any]:
    llm = _create_llm()
    prompts = [
        prompt,
        (
            f"{prompt}\n\n"
            "上一次的输出未通过 JSON 结构校验。请严格按要求返回单个 JSON 对象，"
            "不要输出 Markdown、解释文字或额外字段。"
        ),
    ]

    for index, current_prompt in enumerate(prompts):
        try:
            response = llm.invoke(
                current_prompt,
                response_format={"type": "json_object"},
            )
            return _validate_result(_safe_json_loads(str(response.content)))
        except LLMCallError:
            if index == len(prompts) - 1:
                raise

    raise LLMCallError()


def optimize_resume(
    resume_text: str,
    job_description: str,
    target_role: str = "",
    output_language: str = "auto",
    harvard_format: bool = False
) -> dict[str, Any]:
    """
    调用大模型生成简历优化结果。失败时只抛出“调用LLM失败”。
    """
    role = target_role.strip() or "目标岗位"
    matched_rules = retrieve_resume_expert_rules(resume_text, max_rules=8)
    expert_rules_text = _expert_rules_prompt(matched_rules)
    prompt = f"""
你是资深就业辅导与简历优化智能体。请基于候选人简历和目标岗位生成可直接展示的简历优化结果。

要求：
1. 不编造候选人没有提供的经历。
2. 输出语言：{output_language}。如果为 auto，请跟随简历主要语言。
3. Harvard 格式要求：{"需要" if harvard_format else "不需要"}。
4. 分数必须是 0-100 整数。
5. 返回严格 JSON，不要 Markdown。

目标岗位：{role}
岗位信息：
{job_description[:3000] or "未提供岗位描述，请做通用求职简历优化。"}

候选人简历：
{resume_text[:6000]}

{expert_rules_text}

JSON 格式：
{{
  "overall_score": 0,
  "keyword_score": 0,
  "structure_score": 0,
  "summary": "100字以内总结",
  "optimized_resume": "完整优化稿或重点段落优化稿",
  "strengths": ["优势1", "优势2", "优势3"],
  "weaknesses": ["待提升1", "待提升2", "待提升3"],
  "rewrite_suggestions": [
    {{
      "section": "项目经历",
      "before": "原写法摘要",
      "after": "建议写法",
      "reason": "修改原因"
    }}
  ],
  "action_items": ["下一步动作1", "下一步动作2", "下一步动作3"]
}}
"""

    try:
        result = _invoke_optimizer_with_retry(prompt)
        return _attach_expert_rules(result, matched_rules)
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()
