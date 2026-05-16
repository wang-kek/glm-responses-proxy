import argparse
import json
import sys
import urllib.error
import urllib.request


TEST_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"


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
        lines = []
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
        return resp.status, lines


def check(label: str, ok: bool, detail: str):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify multimodal /v1/responses proxy behavior")
    parser.add_argument("--base-url", required=True, help="Proxy base URL, for example http://127.0.0.1:8080")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    ok = True
    status, payload = request_json(
        args.base_url,
        args.api_key,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "请用一句话描述这张图片里是什么。"},
                        {"type": "input_image", "image_url": TEST_IMAGE_URL},
                    ],
                }
            ],
            "max_output_tokens": 128,
        },
    )
    output_text = payload.get("output_text") or ""
    ok &= check("responses multimodal non-stream", status == 200 and bool(output_text), f"status={status} output_text_len={len(output_text)}")

    status, lines = request_stream(
        args.base_url,
        args.api_key,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "请描述图像内容并指出主要动物。"},
                        {"type": "input_image", "image_url": TEST_IMAGE_URL},
                    ],
                }
            ],
            "stream": True,
            "max_output_tokens": 128,
        },
    )
    has_text_delta = any("response.output_text.delta" in line for line in lines)
    has_completed = any("response.completed" in line for line in lines)
    has_done = any("[DONE]" in line for line in lines)
    ok &= check("responses multimodal stream", status == 200 and has_text_delta and has_completed and has_done, f"status={status} text_delta={has_text_delta} completed={has_completed} done={has_done}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
