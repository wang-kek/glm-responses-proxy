import argparse
import json
import sys
import urllib.error
import urllib.request


TEST_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"


def post_json(base_url: str, body: dict):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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


def post_stream(base_url: str, body: dict):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
    parser = argparse.ArgumentParser(description="Verify a multimodal vLLM upstream with image input")
    parser.add_argument("--base-url", required=True, help="Upstream /v1 base URL")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    ok = True
    status, payload = post_json(
        args.base_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请用一句话描述这张图片里是什么。"},
                        {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                    ],
                }
            ],
            "max_tokens": 128,
        },
    )
    text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    ok &= check("multimodal non-stream", status == 200 and bool(text), f"status={status} text_len={len(text)}")

    status, lines = post_stream(
        args.base_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述图像内容并指出主要动物。"},
                        {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                    ],
                }
            ],
            "stream": True,
            "max_tokens": 128,
        },
    )
    has_text_delta = any('"content"' in line for line in lines)
    has_done = any("[DONE]" in line for line in lines)
    ok &= check("multimodal stream", status == 200 and has_text_delta and has_done, f"status={status} text_delta={has_text_delta} done={has_done}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
