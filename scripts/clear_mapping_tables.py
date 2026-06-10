from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

TABLES_IN_DELETE_ORDER = [
    "mapping_results",
    "student_ability_profiles",
    "student_courses",
    "job_ability_relations",
    "course_ability_relations",
    "courses",
    "ability_tags",
]


def load_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    if not ENV_FILE.exists():
        raise RuntimeError("未找到 .env 文件，也未读取到 DATABASE_URL 环境变量。")

    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(".env 文件中未配置 DATABASE_URL。")


def get_table_count(connection, table_name: str) -> int:
    return int(connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)


def clear_mapping_tables() -> None:
    engine = create_engine(load_database_url(), pool_pre_ping=True)

    with engine.begin() as connection:
        before_counts = {
            table_name: get_table_count(connection, table_name)
            for table_name in TABLES_IN_DELETE_ORDER
        }

        for table_name in TABLES_IN_DELETE_ORDER:
            connection.execute(text(f"DELETE FROM {table_name}"))

        after_counts = {
            table_name: get_table_count(connection, table_name)
            for table_name in TABLES_IN_DELETE_ORDER
        }

    print("已清空课程-能力-岗位智能映射相关表。")
    for table_name in TABLES_IN_DELETE_ORDER:
        print(f"{table_name}: {before_counts[table_name]} -> {after_counts[table_name]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="清空课程-能力-岗位智能映射相关表中的演示数据。"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行清空操作。"
    )
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("该脚本会清空新增映射表数据。确认执行请添加 --yes。")

    clear_mapping_tables()


if __name__ == "__main__":
    main()
