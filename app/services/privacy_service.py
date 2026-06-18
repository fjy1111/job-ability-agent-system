from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import HTTPException, Request


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260_000
ROLE_LABELS = {"student": "学生", "school": "学校", "enterprise": "企业"}
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LOCK = Lock()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def hash_password(password: str) -> str:
    """使用 PBKDF2-SHA256 保存密码，格式长度兼容现有 varchar(100)。"""
    salt = secrets.token_urlsafe(12)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    )
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt}${encoded}"


def verify_password(password: str, stored_password: str) -> tuple[bool, bool]:
    """返回（是否正确，是否为需要升级的旧明文密码）。"""
    stored_password = stored_password or ""
    if not stored_password.startswith(f"{PASSWORD_SCHEME}$"):
        return hmac.compare_digest(password, stored_password), True

    try:
        _, iterations_text, salt, expected = stored_password.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations_text),
        )
        actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    except (TypeError, ValueError):
        return False, False
    return hmac.compare_digest(actual, expected), False


def institution_role_access_error(role: str, access_code: str) -> str:
    """学校与企业身份必须通过服务端环境变量中的机构访问码授权。"""
    if role == "student":
        return ""
    env_name = "SCHOOL_ROLE_ACCESS_CODE" if role == "school" else "ENTERPRISE_ROLE_ACCESS_CODE"
    expected = os.getenv(env_name, "").strip()
    role_label = ROLE_LABELS.get(role, role)
    if not expected:
        return f"系统尚未配置{role_label}机构访问码，请联系管理员。"
    if not hmac.compare_digest((access_code or "").strip(), expected):
        return f"{role_label}机构访问码错误。"
    return ""


def get_or_create_csrf_token(request: Request) -> str:
    token = str(request.session.get("csrf_token") or "")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def require_valid_csrf(request: Request, submitted_token: str) -> None:
    expected = str(request.session.get("csrf_token") or "")
    if not expected or not hmac.compare_digest(expected, (submitted_token or "").strip()):
        raise HTTPException(status_code=403, detail="安全校验失败，请刷新页面后重试。")


def enforce_collaboration_rate_limit(
    request: Request,
    action: str,
    limit: int,
    window_seconds: int = 60,
) -> None:
    """对敏感协同操作做进程内限流，降低批量抓取与暴力提交风险。"""
    user_part = str(request.session.get("user_id") or "anonymous")
    client_part = request.client.host if request.client else "unknown"
    key = f"{action}:{user_part}:{client_part}"
    now = time.monotonic()
    cutoff = now - window_seconds
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="访问过于频繁，请稍后再试。")
        bucket.append(now)


def candidate_privacy_label(record: Any) -> str:
    fingerprint = (getattr(record, "resume_hash", "") or hashlib.sha256(
        f"{getattr(record, 'id', '')}:{getattr(record, 'student_user_id', '')}".encode("utf-8")
    ).hexdigest())[:6].upper()
    return f"候选人-{fingerprint}"


def redact_sensitive_resume_text(text: str, student_name: str = "") -> str:
    """企业端最小化披露：保留能力证据，去除直接身份标识。"""
    value = str(text or "")
    if student_name.strip():
        value = value.replace(student_name.strip(), "[姓名已脱敏]")
    patterns = (
        (r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[手机号已脱敏]"),
        (r"(?<![\w.%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])", "[邮箱已脱敏]"),
        (r"(?<!\d)\d{17}[0-9Xx](?!\d)", "[身份证号已脱敏]"),
        (r"(?im)^\s*(?:家庭住址|现居地址|通信地址|住址)\s*[:：].*$", "地址：[已脱敏]"),
        (r"(?im)^\s*(?:微信|wechat|qq)\s*[:：].*$", "联系方式：[已脱敏]"),
    )
    for pattern, replacement in patterns:
        value = re.sub(pattern, replacement, value)
    return value
