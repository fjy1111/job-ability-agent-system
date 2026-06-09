from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.main import DATABASE_URL, JobKnowledgeRecord, Base  # noqa: E402


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_items(value: str) -> list[str]:
    text_value = clean(value)
    if not text_value:
        return []
    items = []
    for part in text_value.replace("；", "、").replace(";", "、").replace(",", "、").replace("，", "、").split("、"):
        part = part.strip()
        if part:
            items.append(part)
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def ensure_extra_columns(engine) -> None:
    """SQLite 演示库用；MySQL/PostgreSQL 建议改用 Alembic 迁移。"""
    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_knowledge_records)"))}
            extra_cols = {
                "company_name": "VARCHAR(100) DEFAULT ''",
                "hiring_city": "VARCHAR(100) DEFAULT ''",
                "educational_requirements": "VARCHAR(200) DEFAULT ''",
                "salary_range": "VARCHAR(100) DEFAULT ''",
            }
            for col, ddl in extra_cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE job_knowledge_records ADD COLUMN {col} {ddl}"))


def import_jobs(xlsx_path: str | Path) -> int:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    ensure_extra_columns(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    df = pd.read_excel(xlsx_path, sheet_name=0).fillna("")
    required_cols = {
        "job_name", "company_name", "hiring_city", "educational_requirements",
        "required_skills", "related_projects", "recommended_courses",
        "recommended_certificates", "salary_range"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少字段: {sorted(missing)}")

    db = SessionLocal()
    inserted = 0
    try:
        for _, row in df.iterrows():
            job_name = clean(row["job_name"])
            company_name = clean(row["company_name"])
            if not job_name:
                continue

            exists = db.query(JobKnowledgeRecord).filter(
                JobKnowledgeRecord.job_name == job_name,
                JobKnowledgeRecord.company_name == company_name,
            ).first()
            if exists:
                continue

            record = JobKnowledgeRecord(
                job_name=job_name,
                company_name=company_name,
                hiring_city=clean(row["hiring_city"]),
                educational_requirements=clean(row["educational_requirements"]),
                required_skills_json=json.dumps(split_items(row["required_skills"]), ensure_ascii=False),
                related_projects_json=json.dumps(split_items(row["related_projects"]), ensure_ascii=False),
                recommended_courses_json=json.dumps(split_items(row["recommended_courses"]), ensure_ascii=False),
                recommended_certificates_json=json.dumps(split_items(row["recommended_certificates"]), ensure_ascii=False),
                salary_range=clean(row["salary_range"]),
            )
            db.add(record)
            inserted += 1
        db.commit()
        return inserted
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "IT 岗位数据.xlsx"
    count = import_jobs(path)
    print(f"导入完成，新增 {count} 条岗位数据。")
