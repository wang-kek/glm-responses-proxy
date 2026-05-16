import argparse
import json
import sys
import urllib.error
import urllib.request


def request_json(base_url: str, api_key: str, body: dict):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore")
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"raw": text}
        return exc.code, payload


def request_stream(base_url: str, api_key: str, body: dict):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, [
            raw.decode("utf-8", errors="ignore").strip()
            for raw in resp
            if raw.decode("utf-8", errors="ignore").strip()
        ]


def check(label: str, ok: bool, detail: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return ok


def extract_output_types(payload: dict):
    return [item.get("type") for item in payload.get("output", [])]


def first_output_item(payload: dict, item_type: str):
    return next((item for item in payload.get("output", []) if item.get("type") == item_type), None)


def verify(base_url: str, api_key: str):
    ok = True

    status, payload = request_json(
        base_url,
        api_key,
        {
            "input": "请简单介绍你自己，并用一句话回答。",
            "max_output_tokens": 512,
        },
    )
    output_types = extract_output_types(payload)
    ok &= check(
        "basic response",
        status == 200 and "reasoning" in output_types,
        f"status={status} output_types={output_types} output_text_present={bool(payload.get('output_text'))}",
    )

    status, payload = request_json(
        base_url,
        api_key,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "北京现在天气怎么样？请调用工具查询。",
                }
            ],
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
            "max_output_tokens": 256,
        },
    )
    output_types = extract_output_types(payload)
    function_call = first_output_item(payload, "function_call")
    ok &= check(
        "tool call response",
        status == 200 and "function_call" in output_types and function_call is not None,
        f"status={status} output_types={output_types} function_name={(function_call or {}).get('name')}",
    )

    call_id = (function_call or {}).get("call_id")
    status, payload = request_json(
        base_url,
        api_key,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "北京现在天气怎么样？请调用工具查询。",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {
                        "city": "北京",
                        "weather": "晴",
                        "temperature_c": 26,
                    },
                },
            ],
            "max_output_tokens": 256,
        },
    )
    output_types = extract_output_types(payload)
    ok &= check(
        "tool follow-up response",
        status == 200 and "message" in output_types and bool(payload.get("output_text")),
        f"status={status} output_types={output_types} output_text_present={bool(payload.get('output_text'))}",
    )

    status, lines = request_stream(
        base_url,
        api_key,
        {
            "input": "请先思考，再用一句话介绍你自己。",
            "stream": True,
            "max_output_tokens": 256,
        },
    )
    has_reasoning_delta = any("response.reasoning_text.delta" in line for line in lines)
    has_reasoning_done = any("response.reasoning_text.done" in line for line in lines)
    has_completed = any("response.completed" in line for line in lines)
    has_done = any("[DONE]" in line for line in lines)
    ok &= check(
        "stream reasoning events",
        status == 200 and has_reasoning_delta and has_reasoning_done and has_completed and has_done,
        f"status={status} reasoning_delta={has_reasoning_delta} reasoning_done={has_reasoning_done} completed={has_completed} done={has_done}",
    )

    status, lines = request_stream(
        base_url,
        api_key,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "北京现在天气怎么样？请调用工具查询。",
                }
            ],
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
            "max_output_tokens": 256,
        },
    )
    has_fc_delta = any("response.function_call_arguments.delta" in line for line in lines)
    has_fc_done = any("response.function_call_arguments.done" in line for line in lines)
    has_tool_item_done = any("response.output_item.done" in line and "\"function_call\"" in line for line in lines)
    has_done = any("[DONE]" in line for line in lines)
    ok &= check(
        "stream tool events",
        status == 200 and has_fc_delta and has_fc_done and has_tool_item_done and has_done,
        f"status={status} fc_delta={has_fc_delta} fc_done={has_fc_done} tool_item_done={has_tool_item_done} done={has_done}",
    )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify a local /v1/responses proxy against GLM-compatible behavior")
    parser.add_argument("--base-url", required=True, help="Proxy base URL, for example http://127.0.0.1:8080")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    success = verify(args.base_url, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
