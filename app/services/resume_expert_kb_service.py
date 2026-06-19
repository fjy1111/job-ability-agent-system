from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULE_FILE_CANDIDATES = (
    PROJECT_ROOT / "data" / "resume_expert_annotations" / "resume_expert_rules.json",
    PROJECT_ROOT / "data" / " resume_expert_annotations" / "resume_expert_rules.json",
)

DEFAULT_RULE_TITLES = (
    "项目经历需要补充个人角色",
    "尽量补充量化结果",
    "技术亮点要写成“技术方案 + 解决问题 + 效果”",
)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_patterns(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_rule(raw_rule: Any) -> dict | None:
    if not isinstance(raw_rule, dict):
        return None

    title = _as_text(raw_rule.get("title"))
    suggestion = _as_text(raw_rule.get("suggestion"))
    if not title or not suggestion:
        return None

    try:
        priority = int(raw_rule.get("priority", 0))
    except (TypeError, ValueError):
        priority = 0

    return {
        "id": _as_text(raw_rule.get("id")),
        "category": _as_text(raw_rule.get("category")),
        "title": title,
        "problem_patterns": _as_patterns(raw_rule.get("problem_patterns")),
        "suggestion": suggestion,
        "example_before": _as_text(raw_rule.get("example_before")),
        "example_after": _as_text(raw_rule.get("example_after")),
        "source_image": _as_text(raw_rule.get("source_image")),
        "priority": priority,
    }


def load_resume_expert_rules() -> list[dict]:
    """
    Load expert resume rules from the JSON knowledge base.

    The primary path follows CODEX.md. The second candidate keeps the current
    workspace usable if an older data directory accidentally contains a
    leading space.
    """
    for rule_file in RULE_FILE_CANDIDATES:
        try:
            raw_rules = json.loads(rule_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(raw_rules, list):
            return []

        rules = [_normalize_rule(rule) for rule in raw_rules]
        return [rule for rule in rules if rule]

    return []


def _default_rules(rules: list[dict], max_rules: int) -> list[dict]:
    by_title = {rule.get("title"): rule for rule in rules}
    defaults = [
        by_title[title]
        for title in DEFAULT_RULE_TITLES
        if title in by_title
    ]
    if defaults:
        return defaults[:max_rules]
    return sorted(rules, key=lambda item: item.get("priority", 0), reverse=True)[:max_rules]


def retrieve_resume_expert_rules(resume_text: str, max_rules: int = 8) -> list[dict]:
    rules = load_resume_expert_rules()
    try:
        limit = max(0, int(max_rules))
    except (TypeError, ValueError):
        limit = 8

    if not rules or limit == 0:
        return []

    text = str(resume_text or "").lower()
    matched_rules: list[dict] = []
    for rule in rules:
        matched_patterns = [
            pattern
            for pattern in rule.get("problem_patterns", [])
            if pattern.lower() in text
        ]
        if matched_patterns:
            matched_rule = dict(rule)
            matched_rule["matched_patterns"] = matched_patterns
            matched_rules.append(matched_rule)

    if not matched_rules:
        return _default_rules(rules, min(limit, 3))

    return sorted(
        matched_rules,
        key=lambda item: (
            item.get("priority", 0),
            len(item.get("matched_patterns", [])),
        ),
        reverse=True,
    )[:limit]
