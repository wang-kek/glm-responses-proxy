import json
import os
import time
from typing import Any


REQUEST_CAPTURE_LOG = ""


def set_capture_log(path: str):
    global REQUEST_CAPTURE_LOG
    REQUEST_CAPTURE_LOG = path or ""


def _summarize_payload(payload: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = repr(payload)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _summarize_headers(headers: dict) -> dict:
    masked = dict(headers)
    auth = masked.get("Authorization") or masked.get("authorization")
    if auth:
        masked["Authorization"] = auth[:16] + "...redacted"
        masked.pop("authorization", None)
    return masked


def _detect_client_label(headers: dict, body: Any) -> str:
    originator = str(headers.get("originator", "")).lower()
    user_agent = str(headers.get("user-agent", "")).lower()
    body_text = _summarize_payload(body, limit=2000).lower()

    if "hi cli" in body_text:
        return "codex_cli_test"
    if "hi codex" in body_text:
        return "codex_desktop_test"
    if "codex-tui" in originator or "codex-tui" in user_agent:
        return "codex_tui_or_desktop"
    return "unknown_client"


def capture_request(route: str, headers: dict, body: Any) -> str:
    if not REQUEST_CAPTURE_LOG:
        return ""

    try:
        log_dir = os.path.dirname(REQUEST_CAPTURE_LOG)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass

    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "route": route,
        "client_label": _detect_client_label(headers, body),
        "originator": headers.get("originator"),
        "user_agent": headers.get("user-agent"),
        "session_id": headers.get("session_id"),
        "thread_id": headers.get("thread_id"),
        "x_client_request_id": headers.get("x-client-request-id"),
        "x_codex_turn_metadata": headers.get("x-codex-turn-metadata"),
        "headers": _summarize_headers(headers),
        "body": body,
    }

    with open(REQUEST_CAPTURE_LOG, "a", encoding="utf-8") as f:
        f.write("===== REQUEST CAPTURE BEGIN =====\n")
        f.write(json.dumps(record, ensure_ascii=False, indent=2))
        f.write("\n===== REQUEST CAPTURE END =====\n")

    return REQUEST_CAPTURE_LOG
