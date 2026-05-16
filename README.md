# GLM Responses Proxy

`glm-responses-proxy` is a small FastAPI service that translates OpenAI-style `/v1/responses` requests into `/v1/chat/completions` requests for GLM-compatible upstream backends.

It is designed for local or internal deployment where the upstream service already exposes an OpenAI-compatible chat API but the client expects the newer Responses API.

Python requirement: 3.8+

## Features

- Proxies `/v1/responses`, `/v1/chat/completions`, and `/v1/models`
- Converts Responses input into chat-completions messages
- Converts chat-completions output back into Responses output
- Supports streaming responses
- Maps reasoning output into Responses events
- Maps function tools, tool call outputs, and tool-call streaming
- Includes debug logging and verification scripts

## Install

### Option 1: install from the directory

```bash
cd glm-responses-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```

### Option 2: install dependencies only

```bash
cd glm-responses-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default startup uses the built-in defaults from the source:

```bash
glm-responses-proxy
```

Or run the module directly:

```bash
python -m glm_responses_proxy
```

Pass an upstream base URL and listen port when needed:

```bash
glm-responses-proxy \
  --base-url http://localhost:8000/v1 \
  --model glm-5.1-fp8 \
  --multimodal-base-url http://localhost:33338/v1 \
  --multimodal-model Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8080 \
  --debug
```

For simple background startup, use:

```bash
./run.sh start
```

`run.sh` starts the app directly from `src/` and does not require `pip install .` first.

For request-diff testing, you can enable a capture log at `testhi.log` only when needed with `./run.sh start --capture` or `./run.sh restart --capture`. This capture path is isolated from the main proxy logic and can be removed later without changing the core protocol handling.

Optional environment variables:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=glm-5.1-fp8 \
MULTIMODAL_BASE_URL=http://localhost:33338/v1 \
MULTIMODAL_MODEL=Qwen/Qwen3-VL-8B-Instruct \
PORT=8080 DEBUG=1 ./run.sh
```

Service management:

```bash
./run.sh start
./run.sh start --capture
./run.sh stop
./run.sh restart
./run.sh restart --capture
./run.sh status
./run.sh logs
./run.sh --help
```

## CLI Options

```text
--base-url   Upstream `/v1` base URL
--model      Text model name
--multimodal-base-url  Multimodal upstream `/v1` base URL
--multimodal-model     Multimodal model name
--host       Listen host
--port       Listen port
--log-level  Logging level, such as INFO or DEBUG
--debug      Enable verbose debug logging
```

## Endpoints

- `GET /`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## Verification

The `scripts/` directory contains two helpers:

- `scripts/verify_glm_capabilities.py`
  Verifies the upstream GLM-compatible `/v1/chat/completions` service.
- `scripts/verify_responses_proxy.py`
  Verifies the local `/v1/responses` proxy behavior.

Examples:

```bash
python scripts/verify_glm_capabilities.py \
  --base-url http://localhost:8000/v1 \
  --api-key YOUR_API_KEY
```

```bash
python scripts/verify_responses_proxy.py \
  --base-url http://127.0.0.1:8080 \
  --api-key YOUR_API_KEY
```

## GLM Coverage

The current proxy has been verified against `GLM-5.1-FP8` on the upstream `/v1/chat/completions` API.

### Verified upstream GLM capabilities

- Basic chat completion
- Reasoning output in non-stream and stream mode
- Function calling
- Tool-call follow-up with `role="tool"`
- JSON mode through `response_format`
- Streaming tool-call argument deltas

### Verified Responses behavior through this proxy

- `POST /v1/responses` basic text requests
- `POST /v1/responses` image input through `input_image`
- Streamed Responses events with `response.created`, `response.in_progress`, `response.completed`, and `data: [DONE]`
- `reasoning` mapped into Responses output items and stream delta events
- `function_call` mapped into Responses output items
- `function_call_output` mapped back into chat `tool` messages for follow-up turns
- Codex-style flat `function` tools mapped into chat-completions tool format
- Codex-style `custom` tools downgraded into regular function tools so upstream schema validation does not fail immediately
- Automatic routing:
  - text-only requests -> text upstream
  - requests containing image parts -> multimodal upstream

### Important limitations

- This is not full OpenAI Responses parity.
- Built-in OpenAI-hosted tools such as `web_search`, `file_search`, `computer_use`, and `code_interpreter` are not natively implemented by a local GLM model. They need separate orchestration and infrastructure.
- `custom` tools are currently downgraded to plain function tools, which helps compatibility but does not preserve full freeform tool semantics.
- Response persistence features such as `previous_response_id`, retrieval of stored response items, and hosted conversation state are not implemented.
- Multimodal support is currently limited to image input routing and image understanding output.
- Current multimodal support is focused on image input via `input_image` and chat `image_url` parts.
- General `input_file` parsing, PDF parsing, and audio input are not fully implemented.
- Event shapes are close to Responses semantics for Codex-style usage, but not guaranteed to match OpenAI behavior in every field.

## Recommendation

If your goal is to cover more of the modern Responses-style agent workflow locally, a practical direction is:

- For text, reasoning, tool calling, and structured output:
  Deploy a Qwen3 or Qwen3-Coder model on vLLM, with the appropriate reasoning parser and tool-call parser.
- For image and UI understanding:
  Add a Qwen2.5-VL model as a second upstream service.
- For audio/video/speech style multimodal work:
  Add a Qwen2.5-Omni class model, but expect extra serving complexity.

### Suggested deployment path

1. Keep `GLM-5.1-FP8` if your main need is Chinese text reasoning plus basic function calls.
2. If you want stronger local agent compatibility for tool-heavy coding workflows, prefer a Qwen3 or Qwen3-Coder deployment on vLLM.
3. If you want to approach broader Responses coverage, split responsibilities:
   - text/tools: Qwen3 or Qwen3-Coder
   - vision: Qwen2.5-VL
   - audio/video: Qwen2.5-Omni

### Practical advice

Even with a stronger local model, full Responses parity usually requires more than a model swap. You will still need:

- a proxy layer that normalizes tool schemas and stream events
- an orchestration layer for hosted-tool equivalents
- storage if you want retrievable responses or server-managed conversation history

## Project Layout

```text
glm-responses-proxy/
├── pyproject.toml
├── README.md
├── run.sh
├── requirements.txt
├── scripts/
│   ├── verify_glm_capabilities.py
│   └── verify_responses_proxy.py
└── src/
    └── glm_responses_proxy/
        ├── __init__.py
        ├── __main__.py
        └── server.py
```

## Notes

- This project keeps the original default upstream URL and port from the prototype.
- For production use, place the proxy behind your own process supervisor and reverse proxy.
- The proxy does not print raw bearer tokens in logs.
