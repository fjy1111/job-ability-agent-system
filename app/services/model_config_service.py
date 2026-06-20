from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_openai import ChatOpenAI

from app.services.llm_errors import LLMCallError


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    display_name: str
    provider_name: str
    description: str
    accent: str
    api_key_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...]
    model_envs: tuple[str, ...]
    default_base_url: str
    default_model: str = ""


MODEL_DEFINITIONS: dict[str, ModelDefinition] = {
    "chatgpt": ModelDefinition(
        key="chatgpt",
        display_name="ChatGPT 5.5",
        provider_name="OpenAI / OpenAI 兼容通道",
        description="适合综合推理、结构化输出与复杂文本任务。实际 API 模型 ID 由 CHATGPT_MODEL 配置。",
        accent="#10a37f",
        api_key_envs=("CHATGPT_API_KEY", "OPENAI_API_KEY"),
        base_url_envs=("CHATGPT_BASE_URL", "OPENAI_BASE_URL"),
        model_envs=("CHATGPT_MODEL", "OPENAI_MODEL"),
        default_base_url="https://api.openai.com/v1",
    ),
    "gemini": ModelDefinition(
        key="gemini",
        display_name="Gemini",
        provider_name="Google Gemini / OpenAI 兼容接口",
        description="适合长文本理解与多模态扩展。实际 API 模型 ID 由 GEMINI_MODEL 配置。",
        accent="#4285f4",
        api_key_envs=("GEMINI_API_KEY",),
        base_url_envs=("GEMINI_BASE_URL",),
        model_envs=("GEMINI_MODEL",),
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "deepseek": ModelDefinition(
        key="deepseek",
        display_name="DeepSeek",
        provider_name="DeepSeek / 当前项目默认通道",
        description="复用项目现有 DeepSeek 配置，适合中文分析、能力画像和岗位精排。",
        accent="#4d6bfe",
        api_key_envs=("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        base_url_envs=("DEEPSEEK_BASE_URL", "LLM_BASE_URL"),
        model_envs=("DEEPSEEK_MODEL", "LLM_MODEL"),
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
}

DEFAULT_MODEL_KEY = os.getenv("DEFAULT_LLM_PROVIDER", "deepseek").strip().lower()
if DEFAULT_MODEL_KEY not in MODEL_DEFINITIONS:
    DEFAULT_MODEL_KEY = "deepseek"

_ACTIVE_MODEL_KEY: ContextVar[str | None] = ContextVar("active_llm_model_key", default=None)


def normalize_model_key(value: str | None) -> str:
    key = (value or "").strip().lower()
    return key if key in MODEL_DEFINITIONS else DEFAULT_MODEL_KEY


def set_active_model_key(value: str | None) -> Token:
    return _ACTIVE_MODEL_KEY.set(normalize_model_key(value))


def reset_active_model_key(token: Token) -> None:
    _ACTIVE_MODEL_KEY.reset(token)


def get_active_model_key() -> str:
    return normalize_model_key(_ACTIVE_MODEL_KEY.get())


def _first_env(names: Iterable[str]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def resolve_model_config(
    model_key: str | None = None,
    *,
    task_name: str = "",
    legacy_task_model_envs: tuple[str, ...] = (),
) -> dict[str, str | bool]:
    key = normalize_model_key(model_key or get_active_model_key())
    definition = MODEL_DEFINITIONS[key]
    task_name = task_name.strip().upper()

    task_model_envs: list[str] = []
    if task_name:
        task_model_envs.append(f"{key.upper()}_{task_name}_MODEL")
    if key == "deepseek":
        task_model_envs.extend(legacy_task_model_envs)

    api_key = _first_env(definition.api_key_envs)
    base_url = _first_env(definition.base_url_envs) or definition.default_base_url
    model = (
        _first_env(task_model_envs)
        or _first_env(definition.model_envs)
        or definition.default_model
    )
    return {
        "key": key,
        "display_name": definition.display_name,
        "provider_name": definition.provider_name,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "configured": bool(api_key and model),
    }


def get_model_options() -> list[dict[str, Any]]:
    """返回可安全发送到浏览器的模型目录，绝不包含 API Key。"""
    options: list[dict[str, Any]] = []
    for definition in MODEL_DEFINITIONS.values():
        config = resolve_model_config(definition.key)
        options.append({
            "key": definition.key,
            "display_name": definition.display_name,
            "provider_name": definition.provider_name,
            "description": definition.description,
            "accent": definition.accent,
            "model": config["model"] or "尚未配置",
            "base_url": config["base_url"],
            "configured": config["configured"],
        })
    return options


def create_configured_chat_model(
    *,
    temperature: float,
    timeout: int,
    max_retries: int,
    task_name: str = "",
    legacy_task_model_envs: tuple[str, ...] = (),
) -> ChatOpenAI:
    if os.getenv("USE_LLM", "true").strip().lower() != "true":
        raise LLMCallError()

    config = resolve_model_config(
        task_name=task_name,
        legacy_task_model_envs=legacy_task_model_envs,
    )
    if not config["configured"]:
        raise LLMCallError()

    kwargs: dict[str, Any] = {
        "model": config["model"],
        "api_key": config["api_key"],
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]
    return ChatOpenAI(**kwargs)
