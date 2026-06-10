#!/usr/bin/env python3
"""
Moka recruiting-site API crawler.

Usage examples:
  export MOKA_API_KEY="your_api_key"
  python scripts/moka_jobs_crawler.py --org-id example --mode campus --keyword 开发
  python scripts/moka_jobs_crawler.py --org-id example --site-id 123 --fetch-detail

The script calls Moka's documented API endpoints. If the target company requires
authorization, use your own API key or bearer token.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth


API_BASE = "https://api.mokahr.com/api-platform/v1"

TECH_JOB_KEYWORDS = [
    "开发", "软件", "后端", "前端", "算法", "测试", "数据", "运维", "安全",
    "嵌入式", "AI", "人工智能", "大模型", "机器学习", "深度学习", "工程师",
]

SKILL_ALIASES = {
    "Java": [r"\bJava\b", "Java语言"],
    "Python": [r"\bPython\b"],
    "C++": [r"\bC\+\+\b", "C/C++"],
    "C语言": ["C语言"],
    "Go": [r"\bGo\b", "Golang"],
    "JavaScript": [r"\bJavaScript\b", r"\bJS\b"],
    "TypeScript": [r"\bTypeScript\b", r"\bTS\b"],
    "HTML": [r"\bHTML\b", "HTML5"],
    "CSS": [r"\bCSS\b", "CSS3"],
    "Vue": [r"\bVue\b", "Vue.js", "Vue3"],
    "React": [r"\bReact\b", "React.js"],
    "Spring Boot": ["Spring Boot", "SpringBoot"],
    "Django": [r"\bDjango\b"],
    "Flask": [r"\bFlask\b"],
    "FastAPI": [r"\bFastAPI\b"],
    "MySQL": [r"\bMySQL\b"],
    "PostgreSQL": [r"\bPostgreSQL\b", r"\bPostgres\b"],
    "Redis": [r"\bRedis\b"],
    "MongoDB": [r"\bMongoDB\b"],
    "SQL": [r"\bSQL\b", "数据库"],
    "Linux": [r"\bLinux\b"],
    "Git": [r"\bGit\b"],
    "Docker": [r"\bDocker\b"],
    "Kubernetes": [r"\bKubernetes\b", r"\bK8s\b"],
    "Shell": [r"\bShell\b", "Shell脚本"],
    "CI/CD": [r"\bCI/CD\b", "持续集成", "持续交付"],
    "机器学习": ["机器学习", "Machine Learning"],
    "深度学习": ["深度学习", "Deep Learning"],
    "PyTorch": [r"\bPyTorch\b"],
    "TensorFlow": [r"\bTensorFlow\b"],
    "OpenCV": [r"\bOpenCV\b"],
    "NLP": [r"\bNLP\b", "自然语言处理"],
    "大模型": ["大模型", "LLM", "AIGC", "生成式AI"],
    "LangChain": [r"\bLangChain\b"],
    "RAG": [r"\bRAG\b", "检索增强生成"],
    "接口测试": ["接口测试"],
    "自动化测试": ["自动化测试", "Selenium", "pytest"],
    "网络安全": ["网络安全", "Web安全", "渗透测试"],
}

COURSE_BY_SKILL = {
    "Java": "Java程序设计",
    "Python": "Python程序设计",
    "C++": "C++程序设计",
    "SQL": "数据库原理",
    "MySQL": "数据库原理",
    "Redis": "数据库系统实践",
    "Linux": "操作系统",
    "Docker": "云计算与容器技术",
    "Kubernetes": "云计算与容器技术",
    "机器学习": "机器学习",
    "深度学习": "深度学习",
    "大模型": "人工智能导论",
    "NLP": "自然语言处理",
    "Vue": "Web前端开发",
    "React": "Web前端开发",
    "自动化测试": "软件测试技术",
    "网络安全": "网络安全",
}

CERT_BY_JOB_KEYWORD = {
    "开发": ["软考程序员", "软件设计师"],
    "算法": ["人工智能训练师", "英语六级"],
    "大模型": ["人工智能训练师"],
    "数据": ["数据分析相关证书", "英语六级"],
    "测试": ["软件评测师"],
    "运维": ["Linux认证", "云计算相关证书"],
    "安全": ["CISP", "网络安全等级保护相关证书"],
}


@dataclass
class CrawlConfig:
    org_id: str
    mode: str
    site_id: str | None
    keyword: str | None
    zhineng_id: str | None
    status: str
    limit: int
    max_pages: int | None
    delay: float
    fetch_detail: bool
    api_key: str | None
    bearer_token: str | None
    output_dir: Path


def clean_html(value: str | None) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


class MokaClient:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "job-ability-agent-system/0.1",
        })
        if config.bearer_token:
            self.session.headers["Authorization"] = f"Bearer {config.bearer_token}"
        elif config.api_key:
            self.session.auth = HTTPBasicAuth(config.api_key, "")

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_jobs_page(self, offset: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "mode": self.config.mode,
            "limit": self.config.limit,
            "offset": offset,
            "status": self.config.status,
        }
        if self.config.site_id:
            params["siteId"] = self.config.site_id
        if self.config.keyword:
            params["keyword"] = self.config.keyword
        if self.config.zhineng_id:
            params["zhinengId"] = self.config.zhineng_id
        return self.get(f"/jobs/{self.config.org_id}", params=params)

    def fetch_job_detail(self, job_id: str | int) -> dict[str, Any]:
        params: dict[str, Any] = {"mode": self.config.mode}
        if self.config.site_id:
            params["siteId"] = self.config.site_id
        return self.get(f"/jobs/{self.config.org_id}/{job_id}", params=params)


def extract_jobs(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    jobs = payload.get("jobs") or payload.get("data") or payload.get("items") or []
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs") or jobs.get("data") or jobs.get("items") or []
    total = payload.get("total") or payload.get("count") or payload.get("totalCount")
    return jobs if isinstance(jobs, list) else [], int(total) if total is not None else None


def normalize_locations(job: dict[str, Any]) -> list[str]:
    locations = job.get("locations") or job.get("location") or []
    if isinstance(locations, str):
        return [locations]
    if not isinstance(locations, list):
        return []
    result = []
    for item in locations:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(item.get("name") or item.get("city") or item.get("value") or "")
    return unique(result)


def normalize_job(job: dict[str, Any], org_id: str) -> dict[str, Any]:
    raw_description = job.get("description") or job.get("jobDescription") or ""
    title = job.get("title") or job.get("name") or job.get("jobName") or ""
    job_id = job.get("id") or job.get("jobId")
    return {
        "source": "moka",
        "org_id": org_id,
        "job_id": str(job_id) if job_id is not None else "",
        "job_name": title,
        "department": get_nested(job, "department", "name") or job.get("departmentName") or "",
        "zhineng": get_nested(job, "zhineng", "name") or get_nested(job, "function", "name") or "",
        "locations": normalize_locations(job),
        "education": job.get("education") or job.get("degree") or "",
        "commitment": job.get("commitment") or job.get("jobType") or "",
        "min_salary": job.get("minSalary"),
        "max_salary": job.get("maxSalary"),
        "min_experience": job.get("minExperience"),
        "max_experience": job.get("maxExperience"),
        "opened_at": job.get("openedAt") or job.get("createdAt") or "",
        "updated_at": job.get("updatedAt") or "",
        "description": clean_html(raw_description),
        "raw": job,
    }


def is_tech_job(job: dict[str, Any]) -> bool:
    text = " ".join([
        job.get("job_name", ""),
        job.get("zhineng", ""),
        job.get("department", ""),
        job.get("description", ""),
    ])
    return any(keyword.lower() in text.lower() for keyword in TECH_JOB_KEYWORDS)


def extract_skills(text: str) -> list[str]:
    matched: list[str] = []
    for skill, patterns in SKILL_ALIASES.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                matched.append(skill)
                break
    return unique(matched)


def infer_projects(job_name: str, skills: list[str]) -> list[str]:
    name = job_name.lower()
    if any(word in job_name for word in ["大模型", "AI", "人工智能"]):
        return ["知识库问答系统", "智能客服系统", "AI应用助手"]
    if any(word in job_name for word in ["算法", "机器学习", "深度学习"]):
        return ["图像分类项目", "文本分类项目", "推荐算法项目"]
    if "前端" in job_name or {"Vue", "React"} & set(skills):
        return ["后台管理系统", "数据可视化大屏", "个人作品集网站"]
    if "测试" in job_name:
        return ["接口测试项目", "自动化测试脚本", "缺陷管理系统"]
    if any(word in job_name for word in ["运维", "DevOps"]):
        return ["自动化部署平台", "服务监控系统", "日志告警系统"]
    if "安全" in job_name:
        return ["漏洞扫描系统", "安全日志分析系统", "Web攻防实验"]
    if "数据" in job_name or {"SQL", "Python"} <= set(skills):
        return ["用户画像分析", "销售数据分析", "数据看板系统"]
    if "java" in name or "后端" in job_name:
        return ["学生管理系统", "电商后台系统", "权限管理系统"]
    return ["课程实践项目", "综合开发项目"]


def infer_courses(skills: list[str]) -> list[str]:
    courses = [COURSE_BY_SKILL[skill] for skill in skills if skill in COURSE_BY_SKILL]
    return unique(courses)[:5] or ["数据结构", "数据库原理", "软件工程"]


def infer_certificates(job_name: str) -> list[str]:
    certs: list[str] = []
    for keyword, values in CERT_BY_JOB_KEYWORD.items():
        if keyword in job_name:
            certs.extend(values)
    return unique(certs)[:4] or ["软考程序员", "英语六级"]


def to_knowledge_record(job: dict[str, Any]) -> dict[str, Any]:
    text = f"{job['job_name']}\n{job['description']}"
    skills = extract_skills(text)
    return {
        "job_name": job["job_name"],
        "required_skills": skills,
        "related_projects": infer_projects(job["job_name"], skills),
        "recommended_courses": infer_courses(skills),
        "recommended_certificates": infer_certificates(job["job_name"]),
        "source_url": f"https://app.mokahr.com/social_apply/{job['org_id']}/{job['job_id']}",
        "source": "moka",
        "crawl_time": datetime.now().isoformat(timespec="seconds"),
    }


def crawl(config: CrawlConfig) -> list[dict[str, Any]]:
    client = MokaClient(config)
    offset = 0
    page_no = 0
    all_jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    while True:
        page_no += 1
        payload = client.fetch_jobs_page(offset)
        jobs, total = extract_jobs(payload)
        if not jobs:
            break

        for raw_job in jobs:
            job_id = str(raw_job.get("id") or raw_job.get("jobId") or "")
            job_payload = raw_job
            if config.fetch_detail and job_id:
                time.sleep(config.delay)
                job_payload = client.fetch_job_detail(job_id)
                if "job" in job_payload and isinstance(job_payload["job"], dict):
                    job_payload = job_payload["job"]

            normalized = normalize_job(job_payload, config.org_id)
            dedupe_key = normalized["job_id"] or normalized["job_name"]
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            if is_tech_job(normalized):
                all_jobs.append(normalized)

        offset += config.limit
        if config.max_pages and page_no >= config.max_pages:
            break
        if total is not None and offset >= total:
            break
        time.sleep(config.delay)

    return all_jobs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "job_id", "job_name", "department", "zhineng", "locations", "education",
        "commitment", "min_salary", "max_salary", "min_experience",
        "max_experience", "opened_at", "updated_at", "description",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key, "") for key in fields}
            item["locations"] = "、".join(row.get("locations") or [])
            writer.writerow(item)


def parse_args() -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Crawl Moka recruiting API jobs.")
    parser.add_argument("--org-id", required=True, help="Moka orgId from recruiting-site URL")
    parser.add_argument("--mode", default="campus", choices=["campus", "social"], help="招聘模式")
    parser.add_argument("--site-id", help="Moka siteId if the recruiting site has one")
    parser.add_argument("--keyword", help="职位关键词，例如 开发、算法、测试")
    parser.add_argument("--zhineng-id", help="职位职能 ID，可用于只抓技术类")
    parser.add_argument("--status", default="open", help="职位状态，默认 open")
    parser.add_argument("--limit", type=int, default=100, help="每页数量")
    parser.add_argument("--max-pages", type=int, help="最多采集页数，调试时建议设置")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数")
    parser.add_argument("--fetch-detail", action="store_true", help="是否逐条请求职位详情")
    parser.add_argument("--api-key", default=os.getenv("MOKA_API_KEY"), help="Basic Auth API key")
    parser.add_argument("--bearer-token", default=os.getenv("MOKA_BEARER_TOKEN"), help="OAuth2 bearer token")
    parser.add_argument("--output-dir", default="data/moka_jobs", help="输出目录")
    args = parser.parse_args()

    return CrawlConfig(
        org_id=args.org_id,
        mode=args.mode,
        site_id=args.site_id,
        keyword=args.keyword,
        zhineng_id=args.zhineng_id,
        status=args.status,
        limit=args.limit,
        max_pages=args.max_pages,
        delay=args.delay,
        fetch_detail=args.fetch_detail,
        api_key=args.api_key,
        bearer_token=args.bearer_token,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = crawl(config)
    knowledge_records = [to_knowledge_record(job) for job in jobs]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_jsonl = config.output_dir / f"moka_jobs_{config.org_id}_{stamp}.jsonl"
    csv_path = config.output_dir / f"moka_jobs_{config.org_id}_{stamp}.csv"
    knowledge_path = config.output_dir / f"job_knowledge_{config.org_id}_{stamp}.json"

    write_jsonl(raw_jsonl, jobs)
    write_csv(csv_path, jobs)
    knowledge_path.write_text(
        json.dumps(knowledge_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"采集到技术类岗位：{len(jobs)}")
    print(f"原始 JSONL：{raw_jsonl}")
    print(f"岗位 CSV：{csv_path}")
    print(f"知识库 JSON：{knowledge_path}")


if __name__ == "__main__":
    main()
