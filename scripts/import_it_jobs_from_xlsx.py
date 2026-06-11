from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.models.database import DATABASE_URL, Base  # noqa: E402
from app.models.job_knowledge import JobKnowledgeRecord  # noqa: E402


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


EXTRA_COLUMNS = {
    "company_name": "VARCHAR(100) DEFAULT ''",
    "hiring_city": "VARCHAR(100) DEFAULT ''",
    "educational_requirements": "VARCHAR(200) DEFAULT ''",
    "salary_range": "VARCHAR(100) DEFAULT ''",
}


def ensure_extra_columns(engine) -> None:
    """补齐旧表缺少的岗位来源字段。"""
    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_knowledge_records)"))}
            for col, ddl in EXTRA_COLUMNS.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE job_knowledge_records ADD COLUMN {col} {ddl}"))
        elif dialect in {"mysql", "mariadb"}:
            database_name = engine.url.database
            rows = conn.execute(
                text(
                    """
                    SELECT COLUMN_NAME
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = :schema_name
                      AND TABLE_NAME = 'job_knowledge_records'
                    """
                ),
                {"schema_name": database_name},
            )
            existing = {row[0] for row in rows}
            for col, ddl in EXTRA_COLUMNS.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE job_knowledge_records ADD COLUMN {col} {ddl}"))


def import_jobs(xlsx_path: str | Path, replace: bool = False) -> int:
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
        if replace:
            db.query(JobKnowledgeRecord).delete()
            db.flush()

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
    parser = argparse.ArgumentParser(description="导入 data/IT岗位数据.xlsx 到 MySQL 岗位知识表。")
    parser.add_argument(
        "xlsx_path",
        nargs="?",
        default=str(ROOT / "data" / "IT岗位数据.xlsx"),
        help="岗位 Excel 文件路径。"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="清空 job_knowledge_records 后重新导入。"
    )
    args = parser.parse_args()

    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("请先在 .env 中配置 DATABASE_URL。")

    path = Path(args.xlsx_path)
    count = import_jobs(path, replace=args.replace)
    print(f"导入完成，新增 {count} 条岗位数据。")
