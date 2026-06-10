from __future__ import annotations

from io import BytesIO
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
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


def extract_resume_text_from_upload(filename: str, file_content: bytes) -> tuple[str, list[str]]:
    """
    从上传文件中提取简历文本。
    当前提供纯文本兜底能力，保证页面和路由可用。
    """
    suffix = Path(filename).suffix.lower()
    warnings: list[str] = []

    if suffix in TEXT_EXTENSIONS:
        return _decode_text(file_content).strip(), warnings

    if suffix in PDF_EXTENSIONS:
        text, parser_warning = _extract_pdf_text(file_content)
        if text.strip():
            return text.strip(), warnings

        warnings.append(parser_warning)
        return "", warnings

    warnings.append("当前演示环境支持 txt / md / csv / pdf 文件解析；Word 可先复制文本到输入框。")
    return "", warnings


def _extract_pdf_text(file_content: bytes) -> tuple[str, str]:
    """
    多解析器兜底提取 PDF 文本。
    优先 pypdf，其次 PyPDF2，再其次 pdfplumber。
    """
    missing_parsers: list[str] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_content))
        text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(reader.pages))
        if text.strip():
            return text, ""
    except ModuleNotFoundError:
        missing_parsers.append("pypdf")
    except Exception as exc:
        return "", f"PDF 解析失败：{type(exc).__name__}，请复制简历文本到输入框。"

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(file_content))
        text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(reader.pages))
        if text.strip():
            return text, ""
    except ModuleNotFoundError:
        missing_parsers.append("PyPDF2")
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(BytesIO(file_content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in _first_pdf_pages(pdf.pages))
        if text.strip():
            return text, ""
    except ModuleNotFoundError:
        missing_parsers.append("pdfplumber")
    except Exception:
        pass

    if missing_parsers:
        return "", "PDF 解析依赖未安装，请在 agent 环境执行：python -m pip install pypdf，然后重启项目。"

    return "", "PDF 未提取到有效文本，可能是扫描件或图片版简历，请复制简历文本到输入框。"


def _pick_keywords(text: str, limit: int = 8) -> list[str]:
    raw_items = (
        text.replace("\n", " ")
        .replace("，", " ")
        .replace(",", " ")
        .replace("、", " ")
        .replace("；", " ")
        .replace(";", " ")
        .split()
    )

    keywords: list[str] = []
    for item in raw_items:
        word = item.strip("：:。.()（）[]【】")
        if len(word) < 2:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= limit:
            break

    return keywords


def optimize_resume(
    resume_text: str,
    job_description: str,
    target_role: str = "",
    output_language: str = "auto",
    harvard_format: bool = False
) -> dict:
    """
    简历优化兜底逻辑。
    后续如果接入大模型，只需要保持返回字段兼容模板即可。
    """
    role = target_role.strip() or "目标岗位"
    resume_keywords = set(_pick_keywords(resume_text, limit=20))
    jd_keywords = _pick_keywords(job_description, limit=12)
    matched_keywords = [item for item in jd_keywords if item in resume_keywords]
    missing_keywords = [item for item in jd_keywords if item not in resume_keywords]

    keyword_score = min(100, 55 + len(matched_keywords) * 6)
    structure_score = 78 if len(resume_text) >= 300 else 62
    overall_score = round(keyword_score * 0.55 + structure_score * 0.45)

    strengths = [
        "已提供可用于岗位匹配的简历正文",
        "可以围绕目标岗位继续增强项目成果表达"
    ]

    if matched_keywords:
        strengths.insert(0, f"简历中已覆盖部分岗位关键词：{'、'.join(matched_keywords[:6])}")

    weaknesses = [
        f"建议补充岗位关键词：{'、'.join(missing_keywords[:6])}" if missing_keywords else "岗位关键词覆盖较完整，可继续强化成果量化",
        "项目经历建议使用“背景-行动-结果”结构描述",
        "可增加可验证成果，如数据指标、用户规模、性能提升或交付物"
    ]

    optimized_resume = (
        f"【{role} 简历优化稿】\n"
        "个人优势：围绕目标岗位梳理专业技能、项目经验与可交付成果，突出与岗位要求的匹配度。\n\n"
        "项目经历写法建议：\n"
        "1. 说明项目背景和本人职责，避免只罗列技术名词。\n"
        "2. 写清使用的工具、框架、数据或业务场景。\n"
        "3. 用结果收尾，例如准确率、响应时间、完成模块、上线效果或沉淀文档。\n\n"
        "求职表达建议：\n"
        f"面向{role}，建议优先呈现与岗位描述最相关的技能、项目和成果。"
    )

    rewrite_suggestions = [
        {
            "section": "项目经历",
            "before": "参与某项目开发，负责部分功能。",
            "after": "负责核心模块设计与实现，完成需求拆解、接口联调和结果验证，并沉淀项目说明文档。",
            "reason": "把职责、行动和成果写完整，招聘方更容易判断能力。"
        },
        {
            "section": "技能描述",
            "before": "熟悉 Python、数据库、前端。",
            "after": "熟悉 Python 数据处理、SQL 查询与 Web 接口开发，能结合项目场景完成数据分析和功能实现。",
            "reason": "技能要和使用场景绑定，避免空泛罗列。"
        }
    ]

    action_items = [
        "把最匹配岗位的项目放到简历前半部分",
        "为每段项目经历补充至少一个可量化结果",
        "根据 JD 补齐 3-5 个高频关键词"
    ]

    warning = ""
    if output_language != "auto":
        warning = "当前兜底优化不区分输出语言；接入大模型后可按该字段生成不同语言版本。"
    if harvard_format:
        warning = "当前兜底优化暂未生成 Harvard 格式完整简历；已保留该选项用于后续扩展。"

    return {
        "overall_score": overall_score,
        "keyword_score": keyword_score,
        "structure_score": structure_score,
        "summary": f"当前简历与{role}存在基础匹配度，建议继续强化岗位关键词、项目结果和成果量化。",
        "optimized_resume": optimized_resume,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "rewrite_suggestions": rewrite_suggestions,
        "action_items": action_items,
        "agent_warning": warning
    }
