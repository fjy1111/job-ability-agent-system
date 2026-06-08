import io
import json
import os
import re
import site
import sys
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional runtime dependency guard
    ChatOpenAI = None


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

LOCAL_SITE_PACKAGES = BASE_DIR / ".venv" / "Lib" / "site-packages"

for package_path in (str(LOCAL_SITE_PACKAGES), site.getusersitepackages()):
    if package_path and package_path not in sys.path:
        sys.path.append(package_path)


TECH_KEYWORDS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "HTML", "CSS",
    "Vue", "React", "Spring Boot", "Spring Cloud", "MyBatis", "FastAPI",
    "Flask", "Django", "MySQL", "Redis", "MongoDB", "PostgreSQL", "SQL",
    "Linux", "Docker", "Kubernetes", "Git", "Nginx", "Tomcat", "RabbitMQ",
    "Kafka", "微服务", "分布式", "接口", "RESTful", "数据结构", "算法",
    "机器学习", "深度学习", "大模型", "LangChain", "LangGraph", "Pandas",
    "NumPy", "数据分析", "自动化测试", "接口测试", "性能优化", "需求分析",
    "项目管理", "用户体验", "产品设计"
]


ROLE_KEYWORDS = [
    "Java后端工程师", "后端开发工程师", "Python后端开发工程师", "前端开发工程师",
    "AI应用开发工程师", "算法工程师", "机器学习工程师", "数据分析师",
    "测试开发工程师", "产品经理", "运维开发工程师"
]


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_docx_text_with_stdlib(content: bytes) -> str:
    """
    Extract visible text from a DOCX file without third-party dependencies.
    DOCX files are ZIP archives containing WordprocessingML XML documents.
    """
    text_parts: list[str] = []

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        xml_names = [
            name for name in archive.namelist()
            if (
                name == "word/document.xml"
                or re.match(r"word/(header|footer|footnotes|endnotes)\d*\.xml$", name)
            )
        ]

        for name in xml_names:
            root = ET.fromstring(archive.read(name))

            for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                paragraph_parts: list[str] = []

                for node in paragraph.iter():
                    tag = node.tag.rsplit("}", 1)[-1]

                    if tag in {"t", "instrText"} and node.text:
                        paragraph_parts.append(node.text)
                    elif tag == "tab":
                        paragraph_parts.append("\t")
                    elif tag in {"br", "cr"}:
                        paragraph_parts.append("\n")

                paragraph_text = "".join(paragraph_parts).strip()

                if paragraph_text:
                    text_parts.append(paragraph_text)

    return _normalize_extracted_text("\n".join(text_parts))


def _decode_pdf_literal_string(raw: bytes) -> bytes:
    output = bytearray()
    index = 0

    while index < len(raw):
        char = raw[index]

        if char != 0x5C:
            output.append(char)
            index += 1
            continue

        index += 1

        if index >= len(raw):
            break

        escaped = raw[index]

        if escaped in b"nrtbf":
            output.append({
                ord("n"): 10,
                ord("r"): 13,
                ord("t"): 9,
                ord("b"): 8,
                ord("f"): 12,
            }[escaped])
            index += 1
        elif escaped in b"()\\":
            output.append(escaped)
            index += 1
        elif escaped in b"\r\n":
            if escaped == 13 and index + 1 < len(raw) and raw[index + 1] == 10:
                index += 2
            else:
                index += 1
        elif 48 <= escaped <= 55:
            octal = bytes([escaped])
            index += 1

            for _ in range(2):
                if index < len(raw) and 48 <= raw[index] <= 55:
                    octal += bytes([raw[index]])
                    index += 1
                else:
                    break

            output.append(int(octal, 8))
        else:
            output.append(escaped)
            index += 1

    return bytes(output)


def _decode_pdf_text_bytes(raw: bytes) -> str:
    if not raw:
        return ""

    candidates: list[str] = []

    if raw.startswith(b"\xfe\xff"):
        candidates.append(raw[2:].decode("utf-16-be", errors="ignore"))
    elif raw.startswith(b"\xff\xfe"):
        candidates.append(raw[2:].decode("utf-16-le", errors="ignore"))

    if len(raw) >= 2 and len(raw) % 2 == 0:
        try:
            utf16_text = raw.decode("utf-16-be", errors="ignore")
            if sum(ch.isprintable() for ch in utf16_text) >= max(1, len(utf16_text) // 2):
                candidates.append(utf16_text)
        except Exception:
            pass

    for encoding in ("utf-8", "gb18030", "cp1252", "latin-1"):
        try:
            candidates.append(raw.decode(encoding, errors="ignore"))
        except Exception:
            continue

    candidates = [
        text for text in candidates
        if text and sum(ch.isprintable() or ch.isspace() for ch in text) >= max(1, len(text) // 2)
    ]

    if not candidates:
        return ""

    return max(candidates, key=lambda item: sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in item))


def _decode_ascii_hex(data: bytes) -> bytes:
    hex_text = data.split(b">", 1)[0]
    hex_text = re.sub(rb"[^0-9A-Fa-f]", b"", hex_text)

    if len(hex_text) % 2:
        hex_text += b"0"

    try:
        return bytes.fromhex(hex_text.decode("ascii"))
    except Exception:
        return b""


def _iter_pdf_stream_payloads(content: bytes) -> list[bytes]:
    payloads: list[bytes] = []

    for match in re.finditer(rb"(<<[\s\S]{0,4096}?>>)\s*stream\r?\n([\s\S]*?)\r?\nendstream", content):
        stream_dict = match.group(1)
        stream_data = match.group(2)

        if b"/FlateDecode" in stream_dict or b"/Fl" in stream_dict:
            try:
                payloads.append(zlib.decompress(stream_data))
                continue
            except Exception:
                pass

        if b"/ASCIIHexDecode" in stream_dict or b"/AHx" in stream_dict:
            decoded = _decode_ascii_hex(stream_data)
            if decoded:
                payloads.append(decoded)
                continue

        if b"/Filter" not in stream_dict:
            payloads.append(stream_data)

    payloads.append(content)
    return payloads


def _extract_pdf_text_basic(content: bytes) -> str:
    """
    Dependency-free PDF text extraction for simple text-based PDFs.
    It handles uncompressed and FlateDecode streams with literal/hex text tokens.
    Complex scanned PDFs or custom CMaps still need a dedicated parser/OCR.
    """
    text_parts: list[str] = []
    literal_pattern = rb"\((?:\\.|[^\\()])*\)"
    hex_pattern = rb"<([0-9A-Fa-f\s]+)>"
    token_pattern = re.compile(literal_pattern + rb"|" + hex_pattern)

    for payload in _iter_pdf_stream_payloads(content):
        for text_object in re.findall(rb"BT([\s\S]*?)ET", payload):
            object_parts: list[str] = []

            for token_match in token_pattern.finditer(text_object):
                token = token_match.group(0)

                if token.startswith(b"("):
                    decoded_bytes = _decode_pdf_literal_string(token[1:-1])
                else:
                    decoded_bytes = _decode_ascii_hex(token[1:])

                decoded_text = _decode_pdf_text_bytes(decoded_bytes)

                if decoded_text:
                    object_parts.append(decoded_text)

            if object_parts:
                text_parts.append("".join(object_parts))

    return _normalize_extracted_text("\n".join(text_parts))


def _extract_pdf_text(content: bytes) -> tuple[str, str]:
    parser_errors: list[str] = []

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if text.strip():
            return _normalize_extracted_text(text), ""
    except Exception as exc:
        parser_errors.append(f"pypdf: {type(exc).__name__}")

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if text.strip():
            return _normalize_extracted_text(text), ""
    except Exception as exc:
        parser_errors.append(f"PyPDF2: {type(exc).__name__}")

    try:
        from pdfminer.high_level import extract_text  # type: ignore

        text = extract_text(io.BytesIO(content))

        if text.strip():
            return _normalize_extracted_text(text), ""
    except Exception as exc:
        parser_errors.append(f"pdfminer: {type(exc).__name__}")

    try:
        import fitz  # type: ignore

        with fitz.open(stream=content, filetype="pdf") as document:
            text = "\n".join(page.get_text("text") for page in document)

        if text.strip():
            return _normalize_extracted_text(text), ""
    except Exception as exc:
        parser_errors.append(f"PyMuPDF: {type(exc).__name__}")

    basic_text = _extract_pdf_text_basic(content)

    if basic_text:
        return (
            basic_text,
            "当前未安装专业 PDF 解析库，已使用内置解析器提取文本；复杂版式、扫描件或中文嵌入字体可能不完整。"
        )

    return (
        "",
        "PDF 文件未提取到可用文本。请确认不是扫描件，或安装 pypdf 后重试；也可以直接粘贴简历文本。"
    )


def extract_resume_text_from_upload(filename: str, content: bytes) -> tuple[str, list[str]]:
    """
    Best-effort text extraction for uploaded resumes.
    TXT/Markdown/CSV are decoded directly. DOCX is parsed with stdlib. PDF uses
    common third-party parsers first and falls back to a lightweight parser.
    """
    warnings: list[str] = []
    suffix = Path(filename or "").suffix.lower()

    if not content:
        return "", warnings

    if suffix in {".txt", ".md", ".csv"}:
        return _normalize_extracted_text(_decode_bytes(content)), warnings

    if suffix == ".pdf":
        text, warning = _extract_pdf_text(content)
        if warning:
            warnings.append(warning)
        return text, warnings

    if suffix == ".docx":
        try:
            text = _extract_docx_text_with_stdlib(content)

            if text:
                return text, warnings

            warnings.append("DOCX 文件未提取到可用文本，建议把简历内容粘贴到文本框。")
            return "", warnings
        except zipfile.BadZipFile:
            warnings.append("DOCX 文件格式异常，请确认上传的是 .docx 文件。")
            return "", warnings
        except ET.ParseError:
            warnings.append("DOCX 内容解析失败，请尝试另存为新的 .docx 后重新上传。")
            return "", warnings
        except Exception as exc:
            warnings.append(f"DOCX 文件解析失败：{type(exc).__name__}。")
            return "", warnings

    warnings.append("暂时只支持 TXT、PDF、DOCX 文件；也可以直接粘贴简历文本。")
    return "", warnings


def safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return {}

    return {}


def _create_llm() -> tuple[Any | None, str]:
    use_llm = os.getenv("USE_LLM", "true").lower() == "true"
    api_key = _clean_text(os.getenv("LLM_API_KEY"))
    base_url = _clean_text(os.getenv("LLM_BASE_URL"))
    model = _clean_text(os.getenv("LLM_MODEL"))

    if not use_llm:
        return None, "USE_LLM=false，当前使用规则版简历优化。"

    if ChatOpenAI is None:
        return None, "当前环境无法加载 langchain_openai，已使用规则版简历优化。"

    if not api_key or not model:
        return None, "未配置 LLM_API_KEY 或 LLM_MODEL，当前使用规则版简历优化。"

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.25,
        timeout=90,
        max_retries=2
    ), ""


def _extract_keywords(text: str) -> list[str]:
    matched = [
        keyword for keyword in TECH_KEYWORDS
        if keyword.lower() in text.lower()
    ]

    english_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9+#.]{1,}\b", text)

    for term in english_terms:
        if len(term) >= 3 and term not in matched:
            matched.append(term)

    return list(dict.fromkeys(matched))[:18]


def _infer_role(job_description: str, target_role: str) -> str:
    if _clean_text(target_role):
        return _clean_text(target_role)

    for role in ROLE_KEYWORDS:
        if role in job_description:
            return role

    match = re.search(r"(?:岗位|职位|招聘)[:：\s]+([^\n，,。；;]{3,24})", job_description)
    if match:
        return match.group(1).strip()

    return "目标岗位"


def _score_resume(resume_text: str, job_keywords: list[str], keyword_hits: list[str]) -> tuple[int, int, int]:
    ratio = len(keyword_hits) / max(len(job_keywords), 1)
    job_match_score = round(48 + ratio * 47)

    length_score = 76 if len(resume_text) >= 500 else 66 if len(resume_text) >= 220 else 55
    has_numbers = bool(re.search(r"\d+[%+]?|[一二三四五六七八九十]个", resume_text))
    ats_score = min(95, length_score + (10 if has_numbers else 0) + min(len(keyword_hits), 6))

    overall_score = round(job_match_score * 0.55 + ats_score * 0.45)
    return overall_score, job_match_score, ats_score


def _fallback_optimization(
    resume_text: str,
    job_description: str,
    target_role: str,
    harvard_format: bool
) -> dict[str, Any]:
    role = _infer_role(job_description, target_role)
    job_keywords = _extract_keywords(job_description)
    keyword_hits = [
        keyword for keyword in job_keywords
        if keyword.lower() in resume_text.lower()
    ]
    keyword_gaps = [
        keyword for keyword in job_keywords
        if keyword.lower() not in resume_text.lower()
    ][:8]

    overall_score, job_match_score, ats_score = _score_resume(
        resume_text,
        job_keywords,
        keyword_hits
    )

    bullet_style = "动作动词 + 任务场景 + 技术方法 + 量化结果" if harvard_format else "岗位关键词 + 任务职责 + 项目成果"
    hit_text = "、".join(keyword_hits[:10]) if keyword_hits else "请补充与岗位直接相关的技能关键词"
    gap_text = "、".join(keyword_gaps[:8]) if keyword_gaps else "当前岗位关键词覆盖较完整"

    optimized_resume = f"""求职方向：{role}

个人优势
- 围绕{role}岗位要求，突出已具备的技能证据：{hit_text}。
- 将项目经历改写为“{bullet_style}”结构，减少泛泛描述，强化职责、技术栈和结果。
- 简历中避免写入尚未掌握的技能；对短板技能可放入学习计划或面试准备清单。

核心技能
- 已匹配岗位关键词：{hit_text}
- 建议补充或强化：{gap_text}

项目经历优化写法
- 负责/参与某业务模块的需求分析、接口设计与功能实现，结合岗位要求补充技术栈、难点和结果指标。
- 将“做过项目”改成“解决了什么问题、用了什么方法、产生了什么结果”，例如性能提升、缺陷减少、交付周期缩短或用户体验改善。
- 每段项目经历建议控制在 3 到 5 条要点，优先展示和{role}最相关的内容。

原始简历内容整理
{resume_text}
"""

    return {
        "summary": f"规则版优化已根据岗位描述识别关键词，并给出面向“{role}”的简历改写方向。",
        "optimized_resume": optimized_resume,
        "overall_score": overall_score,
        "job_match_score": job_match_score,
        "ats_score": ats_score,
        "strengths": [
            "已保留原始简历信息，避免编造经历。",
            "优先围绕岗位关键词重组技能和项目表达。",
            "建议使用可量化结果提升项目经历可信度。"
        ],
        "keyword_hits": keyword_hits,
        "keyword_gaps": keyword_gaps,
        "rewrite_suggestions": [
            {
                "section": "核心技能",
                "before": "技能罗列较分散时，招聘方难以快速判断匹配度。",
                "after": f"把与{role}相关的技能放在前面，并按语言、框架、数据库、工具分类。",
                "reason": "提升 ATS 关键词命中率和人工筛选效率。"
            },
            {
                "section": "项目经历",
                "before": "负责模块开发、调试和维护等表述较常见。",
                "after": "补充业务场景、技术方案、个人贡献和可验证结果。",
                "reason": "让经历从职责描述变成能力证据。"
            },
            {
                "section": "岗位匹配",
                "before": "简历未覆盖的岗位关键词容易被系统筛掉。",
                "after": f"优先补齐或学习：{gap_text}。",
                "reason": "减少岗位要求与简历表达之间的显性差距。"
            }
        ],
        "action_items": [
            "为每段项目经历补充 1 到 2 个量化指标。",
            "把最匹配目标岗位的项目放到更靠前位置。",
            "检查简历是否包含岗位描述中的核心技术词。"
        ],
        "used_llm": False,
        "agent_warning": ""
    }


def _normalize_result(data: dict[str, Any], fallback: dict[str, Any], used_llm: bool, warning: str) -> dict[str, Any]:
    result = fallback.copy()
    result.update({
        key: value
        for key, value in data.items()
        if value not in (None, "", [])
    })

    for score_key in ("overall_score", "job_match_score", "ats_score"):
        try:
            result[score_key] = max(0, min(100, int(result.get(score_key, fallback[score_key]))))
        except (TypeError, ValueError):
            result[score_key] = fallback[score_key]

    for list_key in ("strengths", "keyword_hits", "keyword_gaps", "rewrite_suggestions", "action_items"):
        if not isinstance(result.get(list_key), list):
            result[list_key] = fallback.get(list_key, [])

    result["used_llm"] = used_llm
    result["agent_warning"] = warning
    return result


def optimize_resume(
    resume_text: str,
    job_description: str,
    target_role: str = "",
    output_language: str = "auto",
    harvard_format: bool = False
) -> dict[str, Any]:
    """
    Optimize a resume against a job description. Uses the configured LLM when
    available and falls back to deterministic keyword-based optimization.
    """
    resume_text = _clean_text(resume_text)
    job_description = _clean_text(job_description)
    fallback = _fallback_optimization(
        resume_text=resume_text,
        job_description=job_description,
        target_role=target_role,
        harvard_format=harvard_format
    )

    llm, setup_warning = _create_llm()
    if llm is None:
        fallback["agent_warning"] = setup_warning
        return fallback

    role = _infer_role(job_description, target_role)
    language_instruction = {
        "auto": "使用与原简历一致的语言输出。",
        "zh": "使用中文输出。",
        "en": "Use English for the optimized resume and analysis."
    }.get(output_language, "使用与原简历一致的语言输出。")
    harvard_instruction = (
        "项目经历请尽量使用哈佛简历格式：动作动词开头，强调个人贡献，补充量化结果。"
        if harvard_format
        else "项目经历请使用清晰、简洁、结果导向的简历表达。"
    )

    prompt = f"""
你是专业的 AI 简历优化顾问，正在为学生求职者优化简历。

重要要求：
1. 不得编造候选人没有提供的经历、证书、公司、奖项、学历或量化结果。
2. 如果需要量化但原文没有数字，只能写“建议补充具体指标”，不能虚构数字。
3. 必须围绕招聘信息优化技能、项目经历、个人优势和 ATS 关键词。
4. {language_instruction}
5. {harvard_instruction}
6. 只输出严格 JSON，不要 Markdown，不要解释文字，不要使用 ```json 代码块。

目标岗位：
{role}

原始简历：
{resume_text}

招聘信息：
{job_description}

请按下面结构输出：
{{
  "summary": "本次优化摘要，80到160字",
  "optimized_resume": "优化后的完整简历文本",
  "overall_score": 0到100的整数,
  "job_match_score": 0到100的整数,
  "ats_score": 0到100的整数,
  "strengths": ["当前简历优势1", "当前简历优势2", "当前简历优势3"],
  "keyword_hits": ["已命中的岗位关键词"],
  "keyword_gaps": ["建议补充或强化的岗位关键词"],
  "rewrite_suggestions": [
    {{
      "section": "简历模块名称",
      "before": "原表达问题或原表达摘要",
      "after": "建议改写方向或示例",
      "reason": "为什么这样改"
    }}
  ],
  "action_items": ["后续行动1", "后续行动2", "后续行动3"]
}}
"""

    try:
        response = llm.invoke(prompt)
        parsed = safe_json_loads(str(response.content))

        if not parsed:
            fallback["agent_warning"] = "大模型返回内容无法解析，已展示规则版优化结果。"
            return fallback

        return _normalize_result(parsed, fallback, True, "")

    except Exception as exc:
        fallback["agent_warning"] = f"大模型调用失败，已展示规则版优化结果：{type(exc).__name__}: {exc}"
        return fallback
