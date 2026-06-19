from __future__ import annotations

import hashlib
import re
from typing import Any


ABILITY_PROFILE_CACHE_VERSION = "ability-profile-v1"

NODE_ORDER = (
    "extract_profile",
    "score_ability",
    "analyze_profile_evidence",
    "diagnose_ability",
    "review_profile",
)

DEFAULT_AGENTS = (
    "画像采集智能体",
    "四维评分智能体",
    "证据抽取智能体",
    "能力归因智能体",
    "质量复核智能体",
)


def build_resume_cache_hash(resume_text: str) -> str:
    """对简历正文做轻量规范化后生成稳定缓存键。"""
    normalized = re.sub(r"\s+", " ", (resume_text or "").strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def attach_resume_cache_metadata(
    agent_result: dict[str, Any],
    resume_hash: str,
) -> dict[str, Any]:
    result = dict(agent_result)
    result["cache_metadata"] = {
        "version": ABILITY_PROFILE_CACHE_VERSION,
        "resume_hash": resume_hash,
    }
    return result


def is_matching_cached_result(
    agent_result: dict[str, Any],
    resume_hash: str,
) -> bool:
    metadata = agent_result.get("cache_metadata")
    return bool(
        isinstance(metadata, dict)
        and metadata.get("version") == ABILITY_PROFILE_CACHE_VERSION
        and metadata.get("resume_hash") == resume_hash
        and agent_result.get("ability_scores")
        and agent_result.get("dimension_insights")
    )


def build_cached_agent_events(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    """把数据库中的完整画像重新拆成与实时智能体相同的五段增量事件。"""
    tool_calls = agent_result.get("tool_calls") or []
    handoffs = agent_result.get("collaboration_log") or []
    workflow_steps = agent_result.get("workflow_steps") or []
    llm_agents = agent_result.get("llm_agents") or []
    tool_limits = (2, 4, 5, 6, len(tool_calls))
    events: list[dict[str, Any]] = []

    data_by_node = {
        "extract_profile": {
            "recognized_skills": agent_result.get("recognized_skills", []),
        },
        "score_ability": {
            "ability_scores": agent_result.get("ability_scores", {}),
            "score_evidence": agent_result.get("score_evidence", {}),
            "recognized_skills": agent_result.get("recognized_skills", []),
        },
        "analyze_profile_evidence": {
            "profile_tags": agent_result.get("profile_tags", []),
            "risk_flags": agent_result.get("risk_flags", []),
            "evidence_cards": agent_result.get("evidence_cards", []),
        },
        "diagnose_ability": {
            "summary": agent_result.get("summary", ""),
            "advantages": agent_result.get("advantages", []),
            "weaknesses": agent_result.get("weaknesses", []),
            "dimension_insights": agent_result.get("dimension_insights", []),
            "development_focus": agent_result.get("development_focus", []),
            "review_findings": agent_result.get("review_findings", []),
        },
        "review_profile": {
            "quality_review": agent_result.get("quality_review", []),
            "review_findings": agent_result.get("review_findings", []),
        },
    }

    for index, node in enumerate(NODE_ORDER):
        step = workflow_steps[index] if index < len(workflow_steps) else {}
        data = dict(data_by_node[node])
        data.update({
            "tool_calls": tool_calls[:tool_limits[index]],
            "collaboration_log": handoffs[:index + 1],
            "llm_agents": llm_agents if index >= 1 else [],
        })
        events.append({
            "type": "agent_step",
            "cache_hit": True,
            "node": node,
            "step": step.get("step", f"0{index + 1}"),
            "title": step.get("agent", DEFAULT_AGENTS[index]),
            "text": step.get("output", "缓存画像数据已读取。"),
            "status": "cached",
            "ability_scores": data.get("ability_scores"),
            "summary": data.get("summary", ""),
            "data": data,
        })

    return events
