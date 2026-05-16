import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://localhost:8000/v1"


def request_json(base_url: str, api_key: str, path: str, body: dict):
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="ignore")


def request_stream(base_url: str, api_key: str, path: str, body: dict):
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        chunks = []
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            chunks.append(line)
        return resp.status, chunks


def check(label: str, ok: bool, detail: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return ok


def verify_chat(base_url: str, api_key: str) -> bool:
    success = True

    status, raw = request_json(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [{"role": "user", "content": "请简单介绍你自己"}],
            "max_tokens": 256,
        },
    )
    data = json.loads(raw)
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    success &= check(
        "chat basic",
        status == 200 and "reasoning" in message,
        f"status={status} content_present={bool(message.get('content'))} reasoning_present={bool(message.get('reasoning'))}",
    )

    status, raw = request_json(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [{"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取指定城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 256,
        },
    )
    data = json.loads(raw)
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    tool_calls = message.get("tool_calls") or []
    success &= check(
        "tool calls",
        status == 200 and bool(tool_calls),
        f"status={status} tool_calls={len(tool_calls)} finish_reason={((data.get('choices') or [{}])[0]).get('finish_reason')}",
    )

    status, raw = request_json(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [
                {"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"},
                {
                    "role": "assistant",
                    "content": "好的，我来查询。",
                    "tool_calls": [
                        {
                            "id": "call_verify",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"北京\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_verify",
                    "content": "{\"city\":\"北京\",\"weather\":\"晴\",\"temperature_c\":26}",
                },
            ],
            "max_tokens": 256,
        },
    )
    data = json.loads(raw)
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    success &= check(
        "tool follow-up",
        status == 200 and bool(message.get("content")),
        f"status={status} answer_len={len(message.get('content') or '')}",
    )

    status, raw = request_json(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [
                {"role": "system", "content": "只输出JSON，不要解释。"},
                {"role": "user", "content": "输出一个JSON对象，包含name和age字段"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
        },
    )
    data = json.loads(raw)
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    success &= check(
        "json mode",
        status == 200 and ("content" in message or "reasoning" in message),
        f"status={status} content_present={bool(message.get('content'))} reasoning_present={bool(message.get('reasoning'))}",
    )

    status, chunks = request_stream(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [{"role": "user", "content": "请先思考，再用一句话介绍你自己"}],
            "stream": True,
            "max_tokens": 128,
        },
    )
    has_reasoning_delta = any('"reasoning"' in line for line in chunks)
    has_done = any("[DONE]" in line for line in chunks)
    success &= check(
        "stream reasoning",
        status == 200 and has_reasoning_delta and has_done,
        f"status={status} reasoning_delta={has_reasoning_delta} done={has_done}",
    )

    status, chunks = request_stream(
        base_url,
        api_key,
        "/chat/completions",
        {
            "model": "glm-5.1-fp8",
            "messages": [{"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取指定城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "tool_stream": True,
            "stream": True,
            "max_tokens": 256,
        },
    )
    has_tool_delta = any('"tool_calls"' in line for line in chunks)
    has_done = any("[DONE]" in line for line in chunks)
    success &= check(
        "stream tool calls",
        status == 200 and has_tool_delta and has_done,
        f"status={status} tool_delta={has_tool_delta} done={has_done}",
    )

    return success


def main():
    parser = argparse.ArgumentParser(description="Verify GLM capability support on a chat-completions endpoint")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    ok = verify_chat(args.base_url.rstrip("/"), args.api_key)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
