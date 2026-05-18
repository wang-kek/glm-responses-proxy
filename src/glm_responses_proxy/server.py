import argparse
import json
import logging
import os
import re
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from .request_capture import REQUEST_CAPTURE_LOG, capture_request, set_capture_log
from .traffic_stats import traffic_stats


# ===================== 配置 =====================
VLLM_BASE_URL = "http://localhost:8000/v1"
GLM_MODEL_NAME = "glm-5.1-fp8"
MULTIMODAL_BASE_URL = ""
MULTIMODAL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8080
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "")
# ===============================================


client: Optional[httpx.AsyncClient] = None
logger = logging.getLogger("glm_proxy")
CONTEXT_TOKEN_SOFT_LIMIT = 180000
CONTEXT_TOKEN_HARD_LIMIT = 200000
TRUNCATED_TEXT_HEAD_CHARS = 12000
TRUNCATED_TEXT_TAIL_CHARS = 36000
SUMMARY_TRIGGER_TOKENS = 150000
SUMMARY_TARGET_MAX_TOKENS = 1200
SUMMARY_KEEP_RECENT_MESSAGES = 4
SUMMARY_TRANSCRIPT_MAX_CHARS = 120000
SUMMARY_MESSAGE_PREFIX = "[glm-responses-proxy summary of earlier conversation]\n"
TOOL_OUTPUT_MAX_CHARS = 12000
TOOL_OUTPUT_HEAD_CHARS = 4000
TOOL_OUTPUT_TAIL_CHARS = 2000
DOCX_EXTRACT_MAX_CHARS = 16000
DOCX_DISCOVERY_MAX_DEPTH = 2
DOCX_DISCOVERY_MAX_CANDIDATES = 12
TOKENIZER_LOAD_ATTEMPTED = False
TOKENIZER_LOAD_ERROR = ""
TOKENIZER = None
TOKENIZER_MODE = "heuristic"
DOCX_PATH_PATTERN = re.compile(r"(?P<path>(?:/|\.?/)?[^\s\"'<>]+\.docx)\b", re.IGNORECASE)
DOCX_NAME_PATTERN = re.compile(r"(?P<name>[^\s/\"'<>]+\.docx)\b", re.IGNORECASE)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="GLM Responses Proxy")
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help="upstream /v1 base url")
    parser.add_argument("--model", default=GLM_MODEL_NAME, help="text model name")
    parser.add_argument("--multimodal-base-url", default=MULTIMODAL_BASE_URL, help="multimodal upstream /v1 base url")
    parser.add_argument("--multimodal-model", default=MULTIMODAL_MODEL_NAME, help="multimodal model name")
    parser.add_argument("--host", default=LISTEN_HOST, help="listen host")
    parser.add_argument("--port", type=int, default=LISTEN_PORT, help="listen port")
    parser.add_argument("--capture-log", default="", help="optional request capture log path for testing")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO/DEBUG")
    parser.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, help="optional local tokenizer path for token estimation")
    return parser.parse_args(argv)


def configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.setLevel(level)


def summarize_payload(payload: Any, limit: int = 600) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = repr(payload)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def summarize_headers(headers: dict) -> dict:
    masked = dict(headers)
    auth = masked.get("Authorization") or masked.get("authorization")
    if auth:
        masked["Authorization"] = auth[:16] + "...redacted"
        masked.pop("authorization", None)
    return masked


def normalize_response_format_for_upstream(fmt: dict) -> Optional[dict]:
    if not isinstance(fmt, dict):
        return None

    fmt_type = fmt.get("type")
    if fmt_type == "json_schema":
        json_schema = fmt.get("json_schema")
        if isinstance(json_schema, dict):
            normalized_json_schema = dict(json_schema)
        else:
            normalized_json_schema = {}
            if fmt.get("name"):
                normalized_json_schema["name"] = fmt.get("name")
            if fmt.get("schema") is not None:
                normalized_json_schema["schema"] = fmt.get("schema")
            if "strict" in fmt:
                normalized_json_schema["strict"] = fmt.get("strict")

        if not isinstance(normalized_json_schema.get("schema"), dict):
            logger.warning("dropping invalid json_schema response_format=%s", summarize_payload(fmt, limit=300))
            return None

        if not normalized_json_schema.get("name"):
            normalized_json_schema["name"] = "structured_output"
        if "strict" not in normalized_json_schema:
            normalized_json_schema["strict"] = bool(fmt.get("strict", False))

        return {
            "type": "json_schema",
            "json_schema": normalized_json_schema,
        }

    if fmt_type == "json_object":
        return {"type": "json_object"}

    if fmt_type == "text":
        return {"type": "text"}

    return None


def get_tokenizer():
    global TOKENIZER_LOAD_ATTEMPTED, TOKENIZER_LOAD_ERROR, TOKENIZER, TOKENIZER_MODE

    if TOKENIZER_LOAD_ATTEMPTED:
        return TOKENIZER

    TOKENIZER_LOAD_ATTEMPTED = True
    if not TOKENIZER_PATH:
        TOKENIZER_MODE = "heuristic"
        return None

    try:
        from transformers import AutoTokenizer

        TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        TOKENIZER_MODE = f"transformers:{TOKENIZER_PATH}"
        logger.info("loaded tokenizer for estimation path=%s", TOKENIZER_PATH)
    except Exception as exc:
        TOKENIZER = None
        TOKENIZER_MODE = "heuristic"
        TOKENIZER_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning("failed to load tokenizer path=%s error=%s; falling back to heuristic", TOKENIZER_PATH, TOKENIZER_LOAD_ERROR)

    return TOKENIZER


def estimate_token_count(text: str) -> int:
    if not text:
        return 0

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    ascii_chars = 0
    non_ascii_chars = 0
    whitespace_chars = 0

    for ch in text:
        if ch.isspace():
            whitespace_chars += 1
        elif ord(ch) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1

    return non_ascii_chars + (ascii_chars + 3) // 4 + (whitespace_chars + 7) // 8


def analyze_chat_body(chat_body: dict) -> dict:
    messages = chat_body.get("messages") or []
    tools = chat_body.get("tools") or []
    multimodal_images = 0
    text_chars = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, str):
                text_chars += len(part)
                continue
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                multimodal_images += 1
            text_chars += len(part.get("text") or "")
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                text_chars += len(image_url.get("url") or "")

    serialized = json.dumps(chat_body, ensure_ascii=False)
    serialized_chars = len(serialized)
    estimated_tokens = estimate_token_count(serialized)

    return {
        "message_count": len(messages),
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "multimodal_image_count": multimodal_images,
        "text_chars": text_chars,
        "serialized_chars": serialized_chars,
        "estimated_tokens": estimated_tokens,
        "max_tokens": chat_body.get("max_tokens"),
        "token_estimator": TOKENIZER_MODE,
    }


def truncate_text_middle(text: str, head_chars: int = TRUNCATED_TEXT_HEAD_CHARS, tail_chars: int = TRUNCATED_TEXT_TAIL_CHARS) -> str:
    if len(text) <= head_chars + tail_chars + 64:
        return text
    return (
        text[:head_chars]
        + "\n...[truncated by glm-responses-proxy to fit model context]...\n"
        + text[-tail_chars:]
    )


def trim_chat_body_to_context_limit(chat_body: dict, route: str) -> Tuple[dict, dict]:
    messages = chat_body.get("messages")
    if not isinstance(messages, list) or not messages:
        return chat_body, {"trimmed": False}

    trimmed_body = dict(chat_body)
    trimmed_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
    trimmed_body["messages"] = trimmed_messages
    removed_messages = 0
    truncated_messages = 0

    def stats() -> dict:
        return analyze_chat_body(trimmed_body)

    current_stats = stats()
    if current_stats["estimated_tokens"] < CONTEXT_TOKEN_SOFT_LIMIT:
        return trimmed_body, {
            "trimmed": False,
            "estimated_tokens": current_stats["estimated_tokens"],
            "message_count": current_stats["message_count"],
        }

    removable_indexes = [
        idx
        for idx, message in enumerate(trimmed_messages)
        if isinstance(message, dict) and message.get("role") != "system"
    ]

    while removable_indexes and current_stats["estimated_tokens"] >= CONTEXT_TOKEN_SOFT_LIMIT:
        remove_index = removable_indexes.pop(0)
        trimmed_messages.pop(remove_index)
        removed_messages += 1
        removable_indexes = [
            idx
            for idx, message in enumerate(trimmed_messages)
            if isinstance(message, dict) and message.get("role") != "system"
        ]
        current_stats = stats()

    if current_stats["estimated_tokens"] >= CONTEXT_TOKEN_SOFT_LIMIT:
        for message in trimmed_messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and len(content) > TRUNCATED_TEXT_HEAD_CHARS + TRUNCATED_TEXT_TAIL_CHARS + 64:
                new_content = truncate_text_middle(content)
                if new_content != content:
                    message["content"] = new_content
                    truncated_messages += 1
                    current_stats = stats()
                    if current_stats["estimated_tokens"] < CONTEXT_TOKEN_SOFT_LIMIT:
                        break

    trim_info = {
        "trimmed": removed_messages > 0 or truncated_messages > 0,
        "removed_messages": removed_messages,
        "truncated_messages": truncated_messages,
        "estimated_tokens": current_stats["estimated_tokens"],
        "message_count": current_stats["message_count"],
        "serialized_chars": current_stats["serialized_chars"],
    }

    if trim_info["trimmed"]:
        logger.warning(
            "%s context trimmed removed_messages=%s truncated_messages=%s estimated_tokens=%s estimator=%s message_count=%s serialized_chars=%s",
            route,
            removed_messages,
            truncated_messages,
            current_stats["estimated_tokens"],
            current_stats["token_estimator"],
            current_stats["message_count"],
            current_stats["serialized_chars"],
        )

    return trimmed_body, trim_info


def reject_if_context_too_large(route: str, chat_body: dict) -> Optional[JSONResponse]:
    stats = analyze_chat_body(chat_body)
    estimated_tokens = stats["estimated_tokens"]

    if estimated_tokens >= CONTEXT_TOKEN_SOFT_LIMIT:
        logger.warning(
            "%s large context estimated_tokens=%s estimator=%s messages=%s tools=%s images=%s serialized_chars=%s max_tokens=%s",
            route,
            estimated_tokens,
            stats["token_estimator"],
            stats["message_count"],
            stats["tool_count"],
            stats["multimodal_image_count"],
            stats["serialized_chars"],
            stats["max_tokens"],
        )

    if estimated_tokens < CONTEXT_TOKEN_HARD_LIMIT:
        return None

    return json_error(
        message=(
            "请求上下文过大，代理估算输入约为 "
            f"{estimated_tokens} tokens，已接近或超过安全上限 {CONTEXT_TOKEN_HARD_LIMIT}。"
            "请缩短 system prompt、减少历史消息，或先对旧上下文做摘要后再重试。"
        ),
        status_code=400,
        error_type="invalid_request_error",
        code="context_length_exceeded",
    )


def format_message_for_summary(message: dict) -> str:
    role = message.get("role", "user")
    content = message.get("content")
    tool_call_id = message.get("tool_call_id")

    if isinstance(content, str):
        text = content
    else:
        text = extract_text_content(content)

    text = truncate_text_middle(text, head_chars=6000, tail_chars=6000)
    prefix = f"{role.upper()}: "
    if role == "tool" and tool_call_id:
        prefix = f"TOOL[{tool_call_id}]: "
    return prefix + text


def build_summary_candidate_indexes(messages: List[dict]) -> List[int]:
    non_system_indexes = [
        idx for idx, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") != "system"
    ]
    if len(non_system_indexes) <= SUMMARY_KEEP_RECENT_MESSAGES:
        return []
    return non_system_indexes[:-SUMMARY_KEEP_RECENT_MESSAGES]


def build_summary_transcript(messages: List[dict], candidate_indexes: List[int]) -> str:
    parts = []
    current_chars = 0

    for idx in candidate_indexes:
        message = messages[idx]
        if not isinstance(message, dict):
            continue
        part = format_message_for_summary(message)
        next_chars = current_chars + len(part) + 2
        if next_chars > SUMMARY_TRANSCRIPT_MAX_CHARS and parts:
            break
        parts.append(part)
        current_chars = next_chars

    return "\n\n".join(parts)


async def summarize_old_messages(
    route: str,
    chat_body: dict,
    headers: dict,
) -> Tuple[dict, dict]:
    messages = chat_body.get("messages")
    if not isinstance(messages, list) or not messages:
        return chat_body, {"summarized": False}

    current_stats = analyze_chat_body(chat_body)
    if current_stats["estimated_tokens"] < SUMMARY_TRIGGER_TOKENS:
        return chat_body, {"summarized": False, "estimated_tokens": current_stats["estimated_tokens"]}

    candidate_indexes = build_summary_candidate_indexes(messages)
    if not candidate_indexes:
        return chat_body, {"summarized": False, "estimated_tokens": current_stats["estimated_tokens"]}

    transcript = build_summary_transcript(messages, candidate_indexes)
    if not transcript.strip():
        return chat_body, {"summarized": False, "estimated_tokens": current_stats["estimated_tokens"]}

    upstream_base_url, selected_model = select_upstream(chat_body)
    summary_request = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Summarize earlier conversation history for continued coding work. "
                    "Preserve user goals, constraints, decisions, file paths, unresolved issues, "
                    "tool results, and any concrete next steps. Be concise but complete."
                ),
            },
            {
                "role": "user",
                "content": transcript,
            },
        ],
        "temperature": 0.2,
        "max_tokens": SUMMARY_TARGET_MAX_TOKENS,
        "stream": False,
    }

    logger.warning(
        "%s context summary start estimated_tokens=%s estimator=%s candidate_messages=%s transcript_chars=%s",
        route,
        current_stats["estimated_tokens"],
        current_stats["token_estimator"],
        len(candidate_indexes),
        len(transcript),
    )

    try:
        resp = await get_client().post(
            f"{upstream_base_url}/chat/completions",
            json=summary_request,
            headers=headers,
        )
        if resp.status_code >= 400:
            logger.warning("%s context summary failed status=%s body=%s", route, resp.status_code, resp.text[:800])
            return chat_body, {
                "summarized": False,
                "estimated_tokens": current_stats["estimated_tokens"],
                "summary_error": f"http_{resp.status_code}",
            }

        payload = resp.json()
        choices = payload.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        summary_text = extract_text_content(message.get("content"))
        summary_text = (summary_text or "").strip()
        if not summary_text:
            logger.warning("%s context summary returned empty output", route)
            return chat_body, {
                "summarized": False,
                "estimated_tokens": current_stats["estimated_tokens"],
                "summary_error": "empty_summary",
            }

        summarized_body = dict(chat_body)
        new_messages: List[dict] = []
        insert_at = candidate_indexes[0]
        kept_recent = [
            msg for idx, msg in enumerate(messages)
            if idx not in set(candidate_indexes)
        ]

        for idx, message in enumerate(messages):
            if idx == insert_at:
                new_messages.append(
                    {
                        "role": "system",
                        "content": SUMMARY_MESSAGE_PREFIX + summary_text,
                    }
                )
            if idx in candidate_indexes:
                continue
            new_messages.append(message)

        summarized_body["messages"] = new_messages
        summarized_stats = analyze_chat_body(summarized_body)
        logger.warning(
            "%s context summary applied removed_messages=%s estimated_tokens_before=%s estimated_tokens_after=%s summary_len=%s kept_recent=%s",
            route,
            len(candidate_indexes),
            current_stats["estimated_tokens"],
            summarized_stats["estimated_tokens"],
            len(summary_text),
            len(kept_recent),
        )
        return summarized_body, {
            "summarized": True,
            "removed_messages": len(candidate_indexes),
            "summary_length": len(summary_text),
            "estimated_tokens_before": current_stats["estimated_tokens"],
            "estimated_tokens_after": summarized_stats["estimated_tokens"],
        }
    except Exception as exc:
        logger.exception("%s context summary exception", route)
        return chat_body, {
            "summarized": False,
            "estimated_tokens": current_stats["estimated_tokens"],
            "summary_error": f"{type(exc).__name__}: {exc}",
        }


async def prepare_chat_body_for_upstream(route: str, chat_body: dict, headers: dict) -> Tuple[dict, Optional[JSONResponse], dict]:
    prepared_body, docx_info = inject_docx_context(chat_body, route)
    initial_stats = analyze_chat_body(prepared_body)
    summary_info = {"summarized": False}

    if initial_stats["estimated_tokens"] >= SUMMARY_TRIGGER_TOKENS:
        prepared_body, summary_info = await summarize_old_messages(route, prepared_body, headers)

    prepared_body, trim_info = trim_chat_body_to_context_limit(prepared_body, route)
    rejection = reject_if_context_too_large(route, prepared_body)
    final_stats = analyze_chat_body(prepared_body)

    return prepared_body, rejection, {
        "initial_stats": initial_stats,
        "docx_info": docx_info,
        "summary_info": summary_info,
        "trim_info": trim_info,
        "final_stats": final_stats,
    }


def usage_to_stats(usage: Optional[dict]) -> Tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=None)
    traffic_stats.start_periodic_log(logger, interval=60)
    yield
    traffic_stats.stop_periodic_log()
    if client is not None:
        await client.aclose()


app = FastAPI(
    title="GLM-5.1 Responses Proxy",
    lifespan=lifespan,
)


def get_client() -> httpx.AsyncClient:
    if client is None:
        raise RuntimeError("HTTP client is not initialized")
    return client


def json_error(
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    code: str = "error",
):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def get_auth_headers_from_request(request: Request) -> Union[dict, JSONResponse]:
    """
    从请求方传入的 Authorization 里取 key，然后原样转发给 vLLM。

    请求方需要传：
    Authorization: Bearer sk-xxxxxx
    """
    authorization = request.headers.get("authorization")

    if not authorization:
        return json_error(
            message="缺少 Authorization 请求头，请使用 Authorization: Bearer <api_key>",
            status_code=401,
            error_type="authentication_error",
            code="missing_api_key",
        )

    if not authorization.lower().startswith("bearer "):
        return json_error(
            message="Authorization 格式错误，请使用 Authorization: Bearer <api_key>",
            status_code=401,
            error_type="authentication_error",
            code="invalid_authorization_format",
        )

    api_key = authorization[7:].strip()

    if not api_key:
        return json_error(
            message="API key 为空",
            status_code=401,
            error_type="authentication_error",
            code="empty_api_key",
        )

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def key_error_response(status_code: int = 401):
    return json_error(
        message="API key 不正确，vLLM 验证未通过",
        status_code=status_code,
        error_type="authentication_error",
        code="invalid_api_key",
    )


def extract_text_content(content: Any) -> str:
    """
    把 Responses API 的 content 转成 Chat Completions 可接受的纯文本。
    这里只处理文本；图片、文件、工具调用先忽略。
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue

            if not isinstance(part, dict):
                continue

            part_type = part.get("type")

            if part_type in ("input_text", "output_text", "text"):
                parts.append(part.get("text", ""))

            elif "text" in part:
                parts.append(part.get("text", ""))

            elif "content" in part:
                parts.append(str(part.get("content", "")))

        return "\n".join([p for p in parts if p])

    return str(content)


def normalize_image_url(value: Any) -> Optional[dict]:
    if isinstance(value, str) and value:
        return {"url": value}
    if isinstance(value, dict) and value.get("url"):
        image_url = {"url": value["url"]}
        if value.get("detail"):
            image_url["detail"] = value["detail"]
        return image_url
    return None


def map_response_part_to_chat_part(part: dict) -> Optional[dict]:
    part_type = part.get("type")

    if part_type in ("input_text", "text", "output_text"):
        return {
            "type": "text",
            "text": part.get("text", ""),
        }

    if part_type in ("input_image", "image_url"):
        image_url = normalize_image_url(part.get("image_url") or part.get("url"))
        if image_url:
            return {
                "type": "image_url",
                "image_url": image_url,
            }

    if part_type == "input_file":
        image_url = normalize_image_url(part.get("file_url") or part.get("image_url") or part.get("url"))
        if image_url:
            return {
                "type": "image_url",
                "image_url": image_url,
            }
        if part.get("file_data"):
            mime_type = part.get("mime_type") or "application/octet-stream"
            return {
                "type": "text",
                "text": f"[input_file mime_type={mime_type}] file_data provided but direct file parsing is not implemented by this proxy.",
            }

    if "text" in part:
        return {
            "type": "text",
            "text": part.get("text", ""),
        }

    return None


def convert_response_content_to_chat_content(content: Any) -> Any:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    chat_parts = []
    text_parts = []

    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue

        if not isinstance(part, dict):
            continue

        chat_part = map_response_part_to_chat_part(part)
        if not chat_part:
            continue

        if chat_part.get("type") == "text":
            text_parts.append(chat_part.get("text", ""))
        else:
            if text_parts:
                chat_parts.append({"type": "text", "text": "\n".join(p for p in text_parts if p)})
                text_parts = []
            chat_parts.append(chat_part)

    if chat_parts:
        if text_parts:
            chat_parts.append({"type": "text", "text": "\n".join(p for p in text_parts if p)})
        return chat_parts

    return "\n".join(p for p in text_parts if p)


def message_has_multimodal_content(message: dict) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("image_url", "input_image"):
            return True
    return False


def chat_body_has_multimodal_input(chat_body: dict) -> bool:
    for message in chat_body.get("messages") or []:
        if isinstance(message, dict) and message_has_multimodal_content(message):
            return True
    return False


def select_upstream(chat_body: dict) -> Tuple[str, str]:
    if chat_body_has_multimodal_input(chat_body):
        if not MULTIMODAL_BASE_URL:
            raise ValueError("multimodal input detected but no multimodal upstream is configured")
        return MULTIMODAL_BASE_URL, MULTIMODAL_MODEL_NAME
    return VLLM_BASE_URL, GLM_MODEL_NAME


def adapt_chat_body_for_upstream(chat_body: dict, upstream_base_url: str) -> dict:
    if not MULTIMODAL_BASE_URL:
        return chat_body

    if upstream_base_url != MULTIMODAL_BASE_URL:
        return chat_body

    if not chat_body_has_multimodal_input(chat_body):
        return chat_body

    adapted = dict(chat_body)
    removed_fields = []

    for key in ("tools", "tool_choice", "tool_stream", "parallel_tool_calls"):
        if key not in adapted:
            continue
        value = adapted.pop(key)
        if key == "tools" and isinstance(value, list):
            removed_fields.append(f"{key}={len(value)}")
        else:
            removed_fields.append(f"{key}={value!r}")

    if removed_fields:
        logger.info(
            "multimodal upstream compatibility mode removed unsupported fields: %s",
            ", ".join(removed_fields),
        )

    return adapted


def get_reasoning_text(message: dict) -> str:
    return (
        message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )


def stringify_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def summarize_tool_output_for_context(output: Any, tool_name: str = "") -> str:
    text = stringify_tool_output(output)
    if len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text

    stripped = text.lstrip()
    looks_like_json = stripped.startswith("{") or stripped.startswith("[")
    line_count = text.count("\n") + 1 if text else 0
    summary_parts = [
        f"tool={tool_name or 'unknown'}",
        f"chars={len(text)}",
        f"lines={line_count}",
        f"looks_like_json={'yes' if looks_like_json else 'no'}",
    ]

    if tool_name == "shell_command":
        summary_parts.append("truncated_by=glm-responses-proxy")

    logger.info(
        "truncated tool output for context tool=%s chars=%s lines=%s looks_like_json=%s",
        tool_name or "unknown",
        len(text),
        line_count,
        looks_like_json,
    )

    return (
        "[tool output truncated for context preservation]\n"
        + "summary: "
        + ", ".join(summary_parts)
        + "\n"
        + "head:\n"
        + text[:TOOL_OUTPUT_HEAD_CHARS]
        + "\n...[truncated by glm-responses-proxy]...\n"
        + "tail:\n"
        + text[-TOOL_OUTPUT_TAIL_CHARS:]
    )


def map_response_format(request_body: dict) -> Optional[dict]:
    response_format = request_body.get("response_format")
    if isinstance(response_format, dict):
        return normalize_response_format_for_upstream(response_format)

    text_config = request_body.get("text")
    if not isinstance(text_config, dict):
        return None

    fmt = text_config.get("format")
    if not isinstance(fmt, dict):
        return None
    return normalize_response_format_for_upstream(fmt)


def extract_docx_text(path: str, max_chars: int = DOCX_EXTRACT_MAX_CHARS) -> Optional[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except Exception:
        return None

    try:
        root = ET.fromstring(xml)
    except Exception:
        return None

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts: List[str] = []
    total_chars = 0

    for paragraph in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        line = "".join(runs).strip()
        if not line:
            continue
        next_total = total_chars + len(line) + 1
        if next_total > max_chars and parts:
            parts.append("...[docx content truncated by glm-responses-proxy]...")
            break
        parts.append(line)
        total_chars = next_total

    if not parts:
        return None

    return "\n".join(parts)


def should_attempt_docx_discovery(messages: List[dict]) -> bool:
    joined_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        joined_parts.append(extract_text_content(message.get("content")))
    joined = "\n".join(part for part in joined_parts if part).lower()
    if ".docx" in joined:
        return True
    return any(keyword in joined for keyword in ("阅读文档", "word文档", "ppt大纲", "md文档", "markdown"))


def discover_nearby_docx_candidates() -> List[str]:
    cwd = os.getcwd()
    roots = [cwd, os.path.dirname(cwd)]
    found = []
    seen = set()

    for root in roots:
        for current_root, dirs, files in os.walk(root):
            rel = os.path.relpath(current_root, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > DOCX_DISCOVERY_MAX_DEPTH:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in {".git", ".run", "__pycache__"}]
            for filename in files:
                if not filename.lower().endswith(".docx"):
                    continue
                path = os.path.join(current_root, filename)
                if path in seen:
                    continue
                seen.add(path)
                found.append(path)
                if len(found) >= DOCX_DISCOVERY_MAX_CANDIDATES:
                    return found
    return found


def inject_docx_context(chat_body: dict, route: str) -> Tuple[dict, dict]:
    messages = chat_body.get("messages")
    if not isinstance(messages, list) or not messages:
        return chat_body, {"injected": False}

    seen_paths = set()
    extracted_docs = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else extract_text_content(content)
        if not text:
            continue

        for match in DOCX_PATH_PATTERN.finditer(text):
            raw_path = match.group("path")
            path = os.path.abspath(os.path.expanduser(raw_path))
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not os.path.isfile(path):
                continue
            extracted = extract_docx_text(path)
            if not extracted:
                continue
            extracted_docs.append((path, extracted))

        for match in DOCX_NAME_PATTERN.finditer(text):
            raw_name = match.group("name")
            path = os.path.abspath(os.path.expanduser(raw_name))
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not os.path.isfile(path):
                continue
            extracted = extract_docx_text(path)
            if not extracted:
                continue
            extracted_docs.append((path, extracted))

    if not extracted_docs and should_attempt_docx_discovery(messages):
        nearby_candidates = discover_nearby_docx_candidates()
        if len(nearby_candidates) == 1:
            path = nearby_candidates[0]
            extracted = extract_docx_text(path)
            if extracted:
                extracted_docs.append((path, extracted))
        elif len(nearby_candidates) > 1:
            logger.info("%s skipped automatic docx discovery candidate_count=%s", route, len(nearby_candidates))

    if not extracted_docs:
        return chat_body, {"injected": False}

    augmented_body = dict(chat_body)
    augmented_messages = list(messages)
    insert_index = 0
    if augmented_messages and isinstance(augmented_messages[0], dict) and augmented_messages[0].get("role") == "system":
        insert_index = 1

    injected_messages = []
    for path, extracted in extracted_docs:
        injected_messages.append(
            {
                "role": "system",
                "content": (
                    "The following DOCX file has been pre-extracted for you. "
                    "Use this text instead of attempting to read the binary .docx directly.\n"
                    f"Source: {path}\n"
                    "Extracted content:\n"
                    f"{extracted}"
                ),
            }
        )

    augmented_messages[insert_index:insert_index] = injected_messages
    augmented_body["messages"] = augmented_messages
    logger.info(
        "%s injected_docx_context files=%s chars=%s",
        route,
        len(extracted_docs),
        sum(len(text) for _, text in extracted_docs),
    )
    return augmented_body, {
        "injected": True,
        "files": [path for path, _ in extracted_docs],
        "chars": sum(len(text) for _, text in extracted_docs),
    }


def get_requested_text_format(request_body: dict) -> Optional[dict]:
    text_config = request_body.get("text")
    if not isinstance(text_config, dict):
        return None
    fmt = text_config.get("format")
    if isinstance(fmt, dict):
        return fmt
    return None


def normalize_structured_output_text(text: str, text_format: Optional[dict]) -> str:
    if not text_format or text_format.get("type") != "json_schema":
        return text

    stripped = (text or "").strip()
    if not stripped:
        return stripped

    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    schema = text_format.get("schema")
    if not isinstance(schema, dict):
        return stripped

    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list) or len(required) != 1:
        return stripped

    key = required[0]
    prop = properties.get(key)
    if not isinstance(prop, dict) or prop.get("type") != "string":
        return stripped

    value = stripped.strip().strip('"')
    max_length = prop.get("maxLength")
    if isinstance(max_length, int) and max_length > 0:
        value = value[:max_length]

    return json.dumps({key: value}, ensure_ascii=False)


def map_reasoning_to_thinking(request_body: dict) -> Optional[dict]:
    thinking = request_body.get("thinking")
    if isinstance(thinking, dict):
        return thinking

    reasoning = request_body.get("reasoning")
    if reasoning is None:
        return None

    if reasoning is False:
        return {"type": "disabled"}

    if isinstance(reasoning, dict):
        if reasoning.get("effort") == "none":
            return {"type": "disabled"}
        return {"type": "enabled"}

    return {"type": "enabled"}


def map_tools_to_chat(tools: Any) -> Optional[List[dict]]:
    if not isinstance(tools, list):
        return None

    mapped_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type")
        if tool_type != "function":
            if tool_type == "custom":
                description_parts = [tool.get("description") or ""]
                tool_format = tool.get("format")
                if isinstance(tool_format, dict):
                    format_summary = summarize_payload(tool_format, limit=400)
                    description_parts.append(f"Input format: {format_summary}")

                mapped_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name") or "custom_tool",
                            "description": "\n\n".join(part for part in description_parts if part),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "Raw tool input string. Preserve the required tool-specific format exactly.",
                                    }
                                },
                                "required": ["input"],
                            },
                            **({"strict": False} if "strict" in tool or tool_type == "custom" else {}),
                        },
                    }
                )
                logger.info(
                    "mapped custom tool to function name=%s",
                    tool.get("name") or "custom_tool",
                )
                continue

            logger.warning("dropping unsupported tool type=%s name=%s", tool_type, tool.get("name"))
            continue

        function = tool.get("function")
        if isinstance(function, dict):
            mapped_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": function.get("name") or tool.get("name") or "",
                        "description": function.get("description") or tool.get("description") or "",
                        "parameters": function.get("parameters") or tool.get("parameters") or {"type": "object", "properties": {}},
                        **({"strict": function.get("strict")} if "strict" in function else ({ "strict": tool.get("strict") } if "strict" in tool else {})),
                    },
                }
            )
            continue

        mapped_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name") or "",
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    **({"strict": tool.get("strict")} if "strict" in tool else {}),
                },
            }
        )

    return mapped_tools


def make_reasoning_item(reasoning_text: str, item_id: Optional[str] = None) -> Optional[dict]:
    if not reasoning_text:
        return None

    return {
        "id": item_id or f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": reasoning_text,
            }
        ],
        "content": [
            {
                "type": "reasoning_text",
                "text": reasoning_text,
            }
        ],
    }


def make_function_call_item(tool_call: dict) -> dict:
    function = tool_call.get("function") or {}
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "call_id": call_id,
        "name": function.get("name") or "",
        "arguments": function.get("arguments") or "",
        "status": "completed",
    }


def make_message_item(text: str, item_id: Optional[str] = None, status: str = "completed") -> dict:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ] if text or status == "completed" else [],
    }


def build_response_output_items(message: dict) -> List[dict]:
    items = []

    reasoning_text = get_reasoning_text(message)
    reasoning_item = make_reasoning_item(reasoning_text)
    if reasoning_item:
        items.append(reasoning_item)

    text = extract_text_content(message.get("content"))
    if text:
        items.append(make_message_item(text))

    for tool_call in message.get("tool_calls") or []:
        items.append(make_function_call_item(tool_call))

    return items


def convert_responses_to_chat(request_body: dict) -> dict:
    """
    把 /v1/responses 请求体转换成 /v1/chat/completions 请求体。
    """
    messages = []
    known_tool_calls: Dict[str, dict] = {}
    saw_aborted_apply_patch = False

    instructions = request_body.get("instructions")
    if instructions:
        messages.append(
            {
                "role": "system",
                "content": extract_text_content(instructions),
            }
        )

    input_data = request_body.get("input", "")

    if isinstance(input_data, str):
        messages.append(
            {
                "role": "user",
                "content": input_data,
            }
        )

    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                messages.append(
                    {
                        "role": "user",
                        "content": item,
                    }
                )
                continue

            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            # 普通 message
            if item_type in (None, "message"):
                role = item.get("role", "user")

                if role == "developer":
                    role = "system"

                # Chat Completions 不支持 developer，统一映射 system
                if role not in ("system", "user", "assistant", "tool"):
                    role = "user"

                content = item.get("content", "")

                messages.append(
                    {
                        "role": role,
                        "content": convert_response_content_to_chat_content(content),
                    }
                )

            elif item_type == "function_call":
                call_id = item.get("call_id") or f"call_{uuid.uuid4().hex}"
                name = item.get("name") or ""
                arguments = item.get("arguments") or ""
                known_tool_calls[call_id] = {
                    "name": name,
                    "arguments": arguments,
                }
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                )

            # Codex / Responses 里可能有 function_call_output
            elif item_type == "function_call_output":
                output = item.get("output", "")
                call_id = item.get("call_id", "")
                tool_name = (known_tool_calls.get(call_id) or {}).get("name") or ""
                content = summarize_tool_output_for_context(output, tool_name=tool_name)
                if content.strip().lower() == "aborted" and tool_name == "apply_patch":
                    saw_aborted_apply_patch = True
                    content = (
                        "Tool execution aborted by client while running apply_patch. "
                        "Do not retry the same patch unchanged. If file writing is unavailable, "
                        "explain the failure briefly or provide the markdown/text result directly."
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    }
                )

    else:
        messages.append(
            {
                "role": "user",
                "content": str(input_data),
            }
        )

    if saw_aborted_apply_patch:
        messages.append(
            {
                "role": "system",
                "content": (
                    "A recent apply_patch call was aborted by the client. "
                    "Prefer a single revised patch or a direct textual result instead of "
                    "repeatedly retrying similar patches."
                ),
            }
        )

    chat_body = {
        "model": GLM_MODEL_NAME,
        "messages": messages,
        "stream": bool(request_body.get("stream", False)),
    }

    # 参数映射
    if "temperature" in request_body:
        chat_body["temperature"] = request_body["temperature"]
    else:
        chat_body["temperature"] = 0.7

    if "top_p" in request_body:
        chat_body["top_p"] = request_body["top_p"]

    if "max_output_tokens" in request_body:
        chat_body["max_tokens"] = request_body["max_output_tokens"]

    if "max_tokens" in request_body:
        chat_body["max_tokens"] = request_body["max_tokens"]

    if "presence_penalty" in request_body:
        chat_body["presence_penalty"] = request_body["presence_penalty"]

    if "frequency_penalty" in request_body:
        chat_body["frequency_penalty"] = request_body["frequency_penalty"]

    if "stop" in request_body:
        chat_body["stop"] = request_body["stop"]

    response_format = map_response_format(request_body)
    if response_format:
        chat_body["response_format"] = response_format

    thinking = map_reasoning_to_thinking(request_body)
    if thinking:
        chat_body["thinking"] = thinking

    mapped_tools = map_tools_to_chat(request_body.get("tools"))
    if mapped_tools is not None:
        chat_body["tools"] = mapped_tools

    if "tool_choice" in request_body:
        chat_body["tool_choice"] = request_body["tool_choice"]

    if "tool_stream" in request_body:
        chat_body["tool_stream"] = bool(request_body["tool_stream"])

    # vLLM 的 Chat Completions 流式可以尝试让它返回 usage
    # 如果 vLLM 版本不支持 stream_options，可以把这几行删掉
    if chat_body["stream"]:
        chat_body["stream_options"] = {
            "include_usage": True,
        }

    return chat_body


def convert_chat_to_responses(chat_response: dict, text_format: Optional[dict] = None) -> dict:
    """
    把 /v1/chat/completions 的非流式返回转换成 /v1/responses 风格返回。
    """
    choices = chat_response.get("choices", [])
    content = ""
    message = {}
    reasoning_text = ""
    output_items = []

    if choices:
        message = choices[0].get("message", {}) or {}
        content = extract_text_content(message.get("content"))
        content = normalize_structured_output_text(content, text_format)
        reasoning_text = get_reasoning_text(message)
        output_items = build_response_output_items(message)

    usage = chat_response.get("usage", {})
    response_id = chat_response.get("id") or f"resp_{uuid.uuid4().hex}"
    created_at = chat_response.get("created", int(time.time()))

    if output_items:
        for item in output_items:
            if item.get("type") == "message":
                message_content = item.get("content") or []
                for part in message_content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        part["text"] = content
    if not output_items and content:
        output_items = [make_message_item(content)]

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": chat_response.get("model") or GLM_MODEL_NAME,
        "output": output_items,
        "parallel_tool_calls": bool(message.get("tool_calls")),
        "previous_response_id": None,
        "reasoning": {
            "effort": None,
            "summary": [
                {
                    "type": "summary_text",
                    "text": reasoning_text,
                }
            ] if reasoning_text else [],
        },
        "service_tier": "default",
        "store": False,
        "temperature": 0.7,
        "text": {
            "format": {
                "type": "text",
            }
        },
        "tool_choice": "none",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1,
        "truncation": "disabled",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "user": None,
        "metadata": {},
        "output_text": content,
    }


async def responses_stream_generator(
    chat_body: dict,
    headers: dict,
    text_format: Optional[dict] = None,
    stream_metrics: Optional[dict] = None,
):
    """
    把 vLLM 的 Chat Completions 流式输出转换成 Responses API 风格 SSE。
    所有退出路径都尽量发送 data: [DONE]。
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    content_index = 0
    output_index = 0
    sequence_number = 0
    full_text = ""
    reasoning_text = ""
    created_at = int(time.time())
    done_sent = False
    response_model = chat_body.get("model", GLM_MODEL_NAME)
    reasoning_item_id = f"rs_{uuid.uuid4().hex}"
    message_item_id = f"msg_{uuid.uuid4().hex}"
    message_item_started = False
    reasoning_item_started = False
    reasoning_output_index: Optional[int] = None
    message_output_index: Optional[int] = None
    next_output_index = 0
    tool_item_order = []
    tool_item_started = set()
    tool_output_index_by_tool_index: Dict[int, int] = {}
    tool_item_id_by_tool_index: Dict[int, str] = {}
    tool_calls_by_index: Dict[int, dict] = {}

    final_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    stream_stats = {
        "reasoning_chunks": 0,
        "text_chunks": 0,
        "tool_argument_chunks": 0,
    }

    def done_event() -> bytes:
        return b"data: [DONE]\n\n"

    def make_response(status: str, output=None, output_text: str = "", error=None):
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "background": False,
            "error": error,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": chat_body.get("max_tokens"),
            "max_tool_calls": None,
            "model": response_model,
            "output": output or [],
            "parallel_tool_calls": bool(tool_calls_by_index),
            "previous_response_id": None,
            "reasoning": {
                "effort": None,
                "summary": [
                    {
                        "type": "summary_text",
                        "text": reasoning_text,
                    }
                ] if reasoning_text else [],
            },
            "service_tier": "default",
            "store": False,
            "temperature": chat_body.get("temperature", 0.7),
            "text": {
                "format": {
                    "type": "text",
                }
            },
            "tool_choice": "none",
            "tools": [],
            "top_logprobs": 0,
            "top_p": chat_body.get("top_p", 1),
            "truncation": "disabled",
            "usage": final_usage,
            "user": None,
            "metadata": {},
            "output_text": output_text,
        }

    def sse(event_type: str, payload: dict):
        nonlocal sequence_number

        payload["type"] = event_type
        payload["sequence_number"] = sequence_number
        payload["event_id"] = f"event_{uuid.uuid4().hex}"

        sequence_number += 1

        return (
            f"event: {event_type}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")

    def error_sse(message: str, code: str = "proxy_error"):
        return sse(
            "error",
            {
                "code": code,
                "message": message,
                "param": None,
            },
        )

    def ensure_message_started():
        nonlocal message_item_started, message_output_index, next_output_index
        if message_item_started:
            return []
        message_item_started = True
        message_output_index = next_output_index
        next_output_index += 1
        return [
            sse(
                "response.output_item.added",
                {
                    "response_id": response_id,
                    "output_index": message_output_index,
                    "item": make_message_item("", item_id=message_item_id, status="in_progress"),
                },
            ),
            sse(
                "response.content_part.added",
                {
                    "response_id": response_id,
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": content_index,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    },
                },
            ),
        ]

    def ensure_reasoning_started():
        nonlocal reasoning_item_started, reasoning_output_index, next_output_index
        if reasoning_item_started:
            return []
        reasoning_item_started = True
        reasoning_output_index = next_output_index
        next_output_index += 1
        return [
            sse(
                "response.output_item.added",
                {
                    "response_id": response_id,
                    "output_index": reasoning_output_index,
                    "item": {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [],
                        "content": [],
                    },
                },
            ),
            sse(
                "response.content_part.added",
                {
                    "response_id": response_id,
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "content_index": 0,
                    "part": {
                        "type": "reasoning_text",
                        "text": "",
                    },
                },
            )
        ]

    def merge_tool_call_delta(delta_tool_call: dict):
        index = delta_tool_call.get("index", 0)
        current = tool_calls_by_index.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            },
        )

        if delta_tool_call.get("id"):
            current["id"] = delta_tool_call["id"]
        if delta_tool_call.get("type"):
            current["type"] = delta_tool_call["type"]

        function_delta = delta_tool_call.get("function") or {}
        if function_delta.get("name"):
            current["function"]["name"] = function_delta["name"]
        if "arguments" in function_delta:
            current["function"]["arguments"] += function_delta.get("arguments") or ""

        return index

    def ensure_tool_item_started(index: int):
        nonlocal next_output_index
        if index in tool_item_started:
            return []

        tool_item_started.add(index)
        tool_item_order.append(index)
        tool_output_index_by_tool_index[index] = next_output_index
        next_output_index += 1
        tool_call = tool_calls_by_index[index]
        item_id = f"fc_stream_{index}"
        tool_item_id_by_tool_index[index] = item_id
        return [
            sse(
                "response.output_item.added",
                {
                    "response_id": response_id,
                    "output_index": tool_output_index_by_tool_index[index],
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.get("id") or f"call_{index}",
                        "name": (tool_call.get("function") or {}).get("name") or "",
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        ]

    def build_completed_output_safe():
        completed_output = []
        completed_output_by_index = {}

        try:
            if reasoning_text and reasoning_output_index is not None:
                reasoning_item = make_reasoning_item(reasoning_text, item_id=reasoning_item_id)
                if reasoning_item:
                    completed_output_by_index[reasoning_output_index] = reasoning_item

            if (full_text or message_item_started) and message_output_index is not None:
                completed_output_by_index[message_output_index] = make_message_item(
                    full_text,
                    item_id=message_item_id,
                    status="completed",
                )

            for index in tool_item_order:
                if index not in tool_output_index_by_tool_index:
                    continue
                tool_item = make_function_call_item(tool_calls_by_index.get(index, {}))
                if index in tool_item_id_by_tool_index:
                    tool_item["id"] = tool_item_id_by_tool_index[index]
                completed_output_by_index[tool_output_index_by_tool_index[index]] = tool_item
        except Exception:
            pass

        for index in sorted(completed_output_by_index):
            completed_output.append(completed_output_by_index[index])

        return completed_output

    try:
        upstream_base_url, selected_model = select_upstream(chat_body)
        chat_body["model"] = selected_model
        chat_body = adapt_chat_body_for_upstream(chat_body, upstream_base_url)
        logger.info(
            "responses stream start upstream=%s headers=%s chat_body=%s",
            f"{upstream_base_url}/chat/completions",
            summarize_headers(headers),
            summarize_payload(chat_body),
        )
        async with get_client().stream(
            "POST",
            f"{upstream_base_url}/chat/completions",
            json=chat_body,
            headers=headers,
        ) as resp:

            if resp.status_code in (401, 403):
                if stream_metrics is not None:
                    stream_metrics["error"] = True
                logger.warning("responses stream upstream auth failed status=%s", resp.status_code)
                yield error_sse(
                    "API key 不正确，vLLM 验证未通过",
                    "invalid_api_key",
                )
                yield done_event()
                done_sent = True
                return

            if resp.status_code >= 400:
                error_bytes = await resp.aread()
                error_text = error_bytes.decode("utf-8", errors="ignore")
                if stream_metrics is not None:
                    stream_metrics["error"] = True
                logger.warning(
                    "responses stream upstream error status=%s body=%s",
                    resp.status_code,
                    error_text[:800],
                )

                yield error_sse(
                    error_text or f"vLLM returned HTTP {resp.status_code}",
                    f"vllm_http_{resp.status_code}",
                )
                yield done_event()
                done_sent = True
                return

            yield sse(
                "response.created",
                {
                    "response": make_response(
                        status="in_progress",
                        output=[],
                        output_text="",
                    )
                },
            )

            yield sse(
                "response.in_progress",
                {
                    "response": make_response(
                        status="in_progress",
                        output=[],
                        output_text="",
                    )
                },
            )

            async for line in resp.aiter_lines():
                line = line.strip()

                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                raw = line[len("data:"):].strip()

                if raw == "[DONE]":
                    logger.debug("responses stream upstream sent [DONE]")
                    break

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if isinstance(usage, dict):
                    final_usage["input_tokens"] = usage.get("prompt_tokens", 0) or final_usage["input_tokens"]
                    final_usage["output_tokens"] = usage.get("completion_tokens", 0) or final_usage["output_tokens"]
                    final_usage["total_tokens"] = usage.get("total_tokens", 0) or final_usage["total_tokens"]

                response_model = data.get("model") or response_model
                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]

                delta = choice.get("delta", {})
                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if reasoning_delta:
                    stream_stats["reasoning_chunks"] += 1
                    reasoning_text += reasoning_delta
                    for event in ensure_reasoning_started():
                        yield event
                    yield sse(
                        "response.reasoning_text.delta",
                        {
                            "response_id": response_id,
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "content_index": 0,
                            "delta": reasoning_delta,
                        },
                    )

                text = delta.get("content") or ""

                if text:
                    stream_stats["text_chunks"] += 1
                    for event in ensure_message_started():
                        yield event
                    full_text += text

                    yield sse(
                        "response.output_text.delta",
                        {
                            "response_id": response_id,
                            "item_id": message_item_id,
                            "output_index": message_output_index,
                            "content_index": content_index,
                            "delta": text,
                            "obfuscation": "",
                        },
                    )

                delta_tool_calls = delta.get("tool_calls") or []
                for delta_tool_call in delta_tool_calls:
                    index = merge_tool_call_delta(delta_tool_call)
                    for event in ensure_tool_item_started(index):
                        yield event
                    function_delta = (delta_tool_call.get("function") or {}).get("arguments")
                    if function_delta is not None:
                        stream_stats["tool_argument_chunks"] += 1
                        yield sse(
                            "response.function_call_arguments.delta",
                            {
                                "response_id": response_id,
                                "item_id": tool_item_id_by_tool_index[index],
                                "output_index": tool_output_index_by_tool_index[index],
                                "call_id": tool_calls_by_index[index].get("id") or f"call_{index}",
                                "delta": function_delta,
                            },
                        )

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    # 不直接 return，让后面统一发送 done/completed
                    pass

        if final_usage["total_tokens"] == 0:
            final_usage["output_tokens"] = max(1, len(full_text) // 4) if full_text else 0
            final_usage["total_tokens"] = final_usage["input_tokens"] + final_usage["output_tokens"]

        full_text = normalize_structured_output_text(full_text, text_format)

        completed_output = []
        completed_output_by_index = {}

        if reasoning_text:
            reasoning_item = make_reasoning_item(reasoning_text, item_id=reasoning_item_id)
            if reasoning_item:
                completed_output_by_index[reasoning_output_index] = reasoning_item
                yield sse(
                    "response.reasoning_text.done",
                    {
                        "response_id": response_id,
                        "item_id": reasoning_item_id,
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "text": reasoning_text,
                    },
                )
                yield sse(
                    "response.content_part.done",
                    {
                        "response_id": response_id,
                        "item_id": reasoning_item_id,
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "reasoning_text",
                            "text": reasoning_text,
                        },
                    },
                )
                yield sse(
                    "response.output_item.done",
                    {
                        "response_id": response_id,
                        "output_index": reasoning_output_index,
                        "item": reasoning_item,
                    },
                )

        if full_text or message_item_started:
            completed_item = make_message_item(full_text, item_id=message_item_id, status="completed")
            completed_output_by_index[message_output_index] = completed_item

            yield sse(
                "response.output_text.done",
                {
                    "response_id": response_id,
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": content_index,
                    "text": full_text,
                    "logprobs": [],
                },
            )

            yield sse(
                "response.content_part.done",
                {
                    "response_id": response_id,
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": content_index,
                    "part": {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                        "logprobs": [],
                    },
                },
            )

            yield sse(
                "response.output_item.done",
                {
                    "response_id": response_id,
                    "output_index": message_output_index,
                    "item": completed_item,
                },
            )

        for index in tool_item_order:
            tool_item = make_function_call_item(tool_calls_by_index[index])
            if index in tool_item_id_by_tool_index:
                tool_item["id"] = tool_item_id_by_tool_index[index]
            completed_output_by_index[tool_output_index_by_tool_index[index]] = tool_item
            yield sse(
                "response.function_call_arguments.done",
                {
                    "response_id": response_id,
                    "item_id": tool_item["id"],
                    "output_index": tool_output_index_by_tool_index[index],
                    "call_id": tool_item["call_id"],
                    "arguments": tool_item["arguments"],
                },
            )
            yield sse(
                "response.output_item.done",
                {
                    "response_id": response_id,
                    "output_index": tool_output_index_by_tool_index[index],
                    "item": tool_item,
                },
            )

        for index in sorted(completed_output_by_index):
            completed_output.append(completed_output_by_index[index])

        yield sse(
            "response.completed",
            {
                "response": make_response(
                    status="completed",
                    output=completed_output,
                    output_text=full_text,
                )
            },
        )

        logger.info(
            "responses stream completed model=%s usage=%s stats=%s output_items=%s output_text_len=%s reasoning_len=%s",
            response_model,
            final_usage,
            stream_stats,
            len(completed_output),
            len(full_text),
            len(reasoning_text),
        )
        if stream_metrics is not None:
            stream_metrics["usage"] = {
                "prompt_tokens": final_usage["input_tokens"],
                "completion_tokens": final_usage["output_tokens"],
                "total_tokens": final_usage["total_tokens"],
            }

        yield done_event()
        done_sent = True

    except Exception as e:
        if stream_metrics is not None:
            stream_metrics["error"] = True
        logger.exception("responses stream proxy error")
        yield error_sse(str(e), "proxy_error")
        try:
            yield sse(
                "response.completed",
                {
                    "response": make_response(
                        status="completed",
                        output=build_completed_output_safe(),
                        output_text=full_text,
                        error={
                            "message": str(e),
                            "type": "proxy_error",
                            "code": "proxy_error",
                        },
                    )
                },
            )
        except Exception:
            pass
        if not done_sent:
            yield done_event()
            done_sent = True


@app.post("/v1/responses")
async def responses_proxy(request: Request):
    headers = get_auth_headers_from_request(request)

    if isinstance(headers, JSONResponse):
        return headers

    try:
        body = await request.json()
    except Exception:
        return json_error(
            message="Invalid JSON body",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_json",
        )

    logger.info(
        "responses request stream=%s headers=%s body=%s",
        bool(body.get("stream", False)),
        summarize_headers(dict(request.headers)),
        summarize_payload(body),
    )
    capture_request("/v1/responses", dict(request.headers), body)
    chat_body = convert_responses_to_chat(body)
    chat_body, context_rejection, prep_info = await prepare_chat_body_for_upstream("/v1/responses", chat_body, headers)
    logger.info(
        "responses context prep initial_tokens=%s final_tokens=%s estimator=%s docx_injected=%s summarized=%s trimmed=%s",
        prep_info["initial_stats"]["estimated_tokens"],
        prep_info["final_stats"]["estimated_tokens"],
        prep_info["final_stats"]["token_estimator"],
        prep_info["docx_info"].get("injected", False),
        prep_info["summary_info"].get("summarized", False),
        prep_info["trim_info"].get("trimmed", False),
    )
    if context_rejection is not None:
        traffic_stats.record(
            endpoint="/v1/responses",
            request_bytes=len(json.dumps(chat_body, ensure_ascii=False).encode("utf-8")),
            error=True,
        )
        return context_rejection
    stream = chat_body.get("stream", False)
    text_format = get_requested_text_format(body)

    if stream:
        upstream_base_url, selected_model = select_upstream(chat_body)
        chat_body["model"] = selected_model
        stream_metrics = {"usage": None, "error": False}

        async def tracked_responses_stream():
            total_bytes = 0
            async for chunk in responses_stream_generator(
                chat_body,
                headers,
                text_format=text_format,
                stream_metrics=stream_metrics,
            ):
                total_bytes += len(chunk) if isinstance(chunk, (bytes, bytearray)) else len(chunk.encode("utf-8")) if isinstance(chunk, str) else 0
                yield chunk
            input_tokens, output_tokens = usage_to_stats(stream_metrics.get("usage"))
            traffic_stats.record(
                endpoint="/v1/responses",
                request_bytes=len(json.dumps(chat_body, ensure_ascii=False).encode("utf-8")),
                response_bytes=total_bytes,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=bool(stream_metrics.get("error")),
            )

        return StreamingResponse(
            tracked_responses_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    upstream_base_url, selected_model = select_upstream(chat_body)
    chat_body["model"] = selected_model
    chat_body = adapt_chat_body_for_upstream(chat_body, upstream_base_url)
    resp = await get_client().post(
        f"{upstream_base_url}/chat/completions",
        json=chat_body,
        headers=headers,
    )
    logger.info("responses non-stream upstream=%s status=%s", upstream_base_url, resp.status_code)

    if resp.status_code in (401, 403):
        traffic_stats.record(
            endpoint="/v1/responses",
            request_bytes=len(json.dumps(chat_body, ensure_ascii=False).encode("utf-8")),
            response_bytes=len(resp.content),
            error=True,
        )
        return key_error_response(resp.status_code)

    if resp.status_code >= 400:
        traffic_stats.record(
            endpoint="/v1/responses",
            request_bytes=len(json.dumps(chat_body, ensure_ascii=False).encode("utf-8")),
            response_bytes=len(resp.content),
            error=True,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type", "application/json"),
        )

    chat_data = resp.json()
    logger.debug("responses non-stream upstream body=%s", summarize_payload(chat_data))
    responses_result = convert_chat_to_responses(chat_data, text_format=text_format)
    input_tokens, output_tokens = usage_to_stats(chat_data.get("usage"))
    traffic_stats.record(
        endpoint="/v1/responses",
        request_bytes=len(json.dumps(chat_body, ensure_ascii=False).encode("utf-8")),
        response_bytes=len(resp.content),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return responses_result


@app.get("/v1/models")
async def models(request: Request):
    headers = get_auth_headers_from_request(request)

    if isinstance(headers, JSONResponse):
        return headers

    upstream_urls = [VLLM_BASE_URL]
    if MULTIMODAL_BASE_URL and MULTIMODAL_BASE_URL != VLLM_BASE_URL:
        upstream_urls.append(MULTIMODAL_BASE_URL)

    merged = []
    seen_ids = set()

    for upstream_url in upstream_urls:
        resp = await get_client().get(
            f"{upstream_url}/models",
            headers=headers,
        )
        logger.info("models upstream=%s status=%s", upstream_url, resp.status_code)

        if resp.status_code in (401, 403):
            traffic_stats.record(
                endpoint="/v1/models",
                response_bytes=len(resp.content),
                error=True,
            )
            return key_error_response(resp.status_code)
        if resp.status_code >= 400:
            traffic_stats.record(
                endpoint="/v1/models",
                response_bytes=len(resp.content),
                error=True,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type", "application/json"),
            )

        payload = resp.json()
        for item in payload.get("data") or []:
            model_id = item.get("id")
            if model_id and model_id not in seen_ids:
                seen_ids.add(model_id)
                merged.append(item)

    traffic_stats.record(endpoint="/v1/models", response_bytes=len(json.dumps(merged, ensure_ascii=False).encode("utf-8")))

    return {
        "object": "list",
        "data": merged,
    }


@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    headers = get_auth_headers_from_request(request)

    if isinstance(headers, JSONResponse):
        return headers

    try:
        body = await request.json()
    except Exception:
        return json_error(
            message="Invalid JSON body",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_json",
        )

    upstream_base_url, selected_model = select_upstream(body)
    body["model"] = selected_model
    logger.info(
        "chat request stream=%s headers=%s body=%s",
        bool(body.get("stream", False)),
        summarize_headers(dict(request.headers)),
        summarize_payload(body),
    )
    capture_request("/v1/chat/completions", dict(request.headers), body)
    body, context_rejection, prep_info = await prepare_chat_body_for_upstream("/v1/chat/completions", body, headers)
    logger.info(
        "chat context prep initial_tokens=%s final_tokens=%s estimator=%s docx_injected=%s summarized=%s trimmed=%s",
        prep_info["initial_stats"]["estimated_tokens"],
        prep_info["final_stats"]["estimated_tokens"],
        prep_info["final_stats"]["token_estimator"],
        prep_info["docx_info"].get("injected", False),
        prep_info["summary_info"].get("summarized", False),
        prep_info["trim_info"].get("trimmed", False),
    )
    if context_rejection is not None:
        traffic_stats.record(
            endpoint="/v1/chat/completions",
            request_bytes=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
            error=True,
        )
        return context_rejection

    # 非流式 chat 代理
    if not body.get("stream", False):
        resp = await get_client().post(
            f"{upstream_base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        logger.info("chat non-stream upstream=%s status=%s", upstream_base_url, resp.status_code)

        if resp.status_code in (401, 403):
            traffic_stats.record(
                endpoint="/v1/chat/completions",
                request_bytes=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
                response_bytes=len(resp.content),
                error=True,
            )
            return key_error_response(resp.status_code)

        input_tokens, output_tokens = usage_to_stats(resp.json().get("usage")) if resp.status_code < 400 else (0, 0)
        traffic_stats.record(
            endpoint="/v1/chat/completions",
            request_bytes=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
            response_bytes=len(resp.content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=resp.status_code >= 400,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type", "application/json"),
        )

    # 流式 chat 代理
    async def chat_stream_generator():
        done_sent = False

        def chat_done():
            return b"data: [DONE]\n\n"

        try:
            logger.info(
                "chat stream start upstream=%s body=%s",
                f"{upstream_base_url}/chat/completions",
                summarize_payload(body),
            )
            async with get_client().stream(
                "POST",
                f"{upstream_base_url}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:

                if resp.status_code in (401, 403):
                    stream_metrics["error"] = True
                    logger.warning("chat stream upstream auth failed status=%s", resp.status_code)
                    error_data = {
                        "error": {
                            "message": "API key 不正确，vLLM 验证未通过",
                            "type": "authentication_error",
                            "code": "invalid_api_key",
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                    yield chat_done()
                    done_sent = True
                    return

                if resp.status_code >= 400:
                    error_bytes = await resp.aread()
                    error_text = error_bytes.decode("utf-8", errors="ignore")
                    stream_metrics["error"] = True
                    logger.warning(
                        "chat stream upstream error status=%s body=%s",
                        resp.status_code,
                        error_text[:800],
                    )
                    error_data = {
                        "error": {
                            "message": error_text,
                            "type": "vllm_error",
                            "code": str(resp.status_code),
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                    yield chat_done()
                    done_sent = True
                    return

                saw_done_from_vllm = False

                async for chunk in resp.aiter_bytes():
                    if b"data: [DONE]" in chunk:
                        saw_done_from_vllm = True
                        done_sent = True

                    yield chunk

                # 有些情况下上游结束了但没有显式 [DONE]，这里补一个
                if not saw_done_from_vllm and not done_sent:
                    yield chat_done()
                    done_sent = True
                logger.info("chat stream completed saw_done_from_vllm=%s", saw_done_from_vllm)

        except Exception as e:
            stream_metrics["error"] = True
            logger.exception("chat stream proxy error")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "proxy_error",
                    "code": "proxy_error",
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
            if not done_sent:
                yield chat_done()
                done_sent = True

    stream_metrics = {"error": False}

    async def tracked_chat_stream():
        total_bytes = 0
        async for chunk in chat_stream_generator():
            total_bytes += len(chunk) if isinstance(chunk, (bytes, bytearray)) else 0
            yield chunk
        traffic_stats.record(
            endpoint="/v1/chat/completions",
            request_bytes=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
            response_bytes=total_bytes,
            error=bool(stream_metrics.get("error")),
        )

    return StreamingResponse(
        tracked_chat_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root():
    return {
        "name": "GLM-5.1 Responses Proxy",
        "status": "ok",
        "vllm_base_url": VLLM_BASE_URL,
        "model": GLM_MODEL_NAME,
        "multimodal_base_url": MULTIMODAL_BASE_URL or None,
        "multimodal_model": MULTIMODAL_MODEL_NAME if MULTIMODAL_BASE_URL else None,
        "endpoints": [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/models",
        ],
        "traffic_stats": traffic_stats.snapshot(),
    }


def main(argv: Optional[List[str]] = None):
    global VLLM_BASE_URL, GLM_MODEL_NAME, MULTIMODAL_BASE_URL, MULTIMODAL_MODEL_NAME, TOKENIZER_PATH

    args = parse_args(argv)
    VLLM_BASE_URL = args.base_url.rstrip("/")
    GLM_MODEL_NAME = args.model
    MULTIMODAL_BASE_URL = args.multimodal_base_url.rstrip("/") if args.multimodal_base_url else ""
    MULTIMODAL_MODEL_NAME = args.multimodal_model
    TOKENIZER_PATH = args.tokenizer or ""
    set_capture_log(args.capture_log)
    configure_logging("DEBUG" if args.debug else args.log_level)
    logger.info(
        "starting proxy upstream=%s text_model=%s multimodal_upstream=%s multimodal_model=%s host=%s port=%s level=%s tokenizer=%s",
        VLLM_BASE_URL,
        GLM_MODEL_NAME,
        MULTIMODAL_BASE_URL or None,
        MULTIMODAL_MODEL_NAME if MULTIMODAL_BASE_URL else None,
        args.host,
        args.port,
        "DEBUG" if args.debug else args.log_level.upper(),
        TOKENIZER_PATH or None,
    )
    if REQUEST_CAPTURE_LOG:
        logger.info("request capture log=%s", REQUEST_CAPTURE_LOG)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
