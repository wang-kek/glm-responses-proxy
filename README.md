# GLM Responses Proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)

**English** | [中文](#中文文档)

A lightweight FastAPI proxy that translates OpenAI-style `/v1/responses` requests into `/v1/chat/completions` requests for GLM-compatible upstream backends (such as vLLM).

It is designed for local or internal deployment where the upstream service already exposes an OpenAI-compatible chat API but the client expects the newer Responses API.

Python requirement: 3.8+

## Features

- Proxies `/v1/responses`, `/v1/chat/completions`, and `/v1/models`
- Converts Responses input into chat-completions messages
- Converts chat-completions output back into Responses output
- Supports streaming responses with proper SSE events
- Maps reasoning output into Responses events
- Maps function tools, tool call outputs, and tool-call streaming
- Automatic routing: text-only requests → text upstream, image requests → multimodal upstream
- Built-in traffic statistics (request count, token usage, data transfer)
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
--base-url             Upstream `/v1` base URL
--model                Text model name
--multimodal-base-url  Multimodal upstream `/v1` base URL
--multimodal-model     Multimodal model name
--host                 Listen host
--port                 Listen port
--log-level            Logging level, such as INFO or DEBUG
--debug                Enable verbose debug logging
```

## Endpoints

- `GET /` — Service info and traffic statistics
- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — Chat completions proxy
- `POST /v1/responses` — Responses API proxy

## Traffic Statistics

The proxy tracks traffic metrics and outputs a summary log every 60 seconds. You can also query live stats from the `GET /` endpoint.

Tracked metrics:

- Total requests per endpoint (`/v1/responses`, `/v1/chat/completions`, `/v1/models`)
- Success and error counts
- Token usage (input tokens, output tokens, total tokens)
- Data transfer (bytes sent to upstream, bytes received from upstream)

Example log output:

```
2026-05-16 15:00:00 INFO glm_proxy.traffic [traffic stats] requests=42 responses_stream=28 responses_nonstream=8 chat_stream=4 chat_nonstream=2 models=0 | errors=1 | input_tokens=12340 output_tokens=5678 total_tokens=18018 | bytes_up=256000 bytes_down=1024000 | uptime=3600s
```

## Verification

The `scripts/` directory contains verification helpers:

- `scripts/verify_glm_capabilities.py` — Verifies the upstream GLM-compatible `/v1/chat/completions` service
- `scripts/verify_responses_proxy.py` — Verifies the local `/v1/responses` proxy behavior
- `scripts/verify_multimodal_upstream.py` — Verifies a multimodal vLLM upstream with image input
- `scripts/verify_multimodal_proxy.py` — Verifies multimodal `/v1/responses` proxy behavior

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
- Automatic routing: text-only requests → text upstream, image requests → multimodal upstream

### Important limitations

- This is not full OpenAI Responses parity.
- OpenAI-hosted tools such as `web_search`, `file_search`, `computer_use`, and `code_interpreter` work through client-side orchestration: the model decides when to call them, the client executes the call, and results are returned via `function_call_output`. GLM correctly outputs tool call format, so these tools function normally when the client supports them. This proxy passes tool calls through without modification.
- `custom` tools are currently downgraded to plain function tools, which helps compatibility with upstream schema validation but does not preserve full freeform tool semantics.
- Response persistence features such as `previous_response_id`, retrieval of stored response items, and hosted conversation state are not implemented.
- Multimodal support is currently limited to image input routing and image understanding output.
- General `input_file` parsing, PDF parsing, and audio input are not fully implemented.
- Event shapes are close to Responses semantics for Codex-style usage, but not guaranteed to match OpenAI behavior in every field.

## Project Layout

```text
glm-responses-proxy/
├── pyproject.toml
├── README.md
├── run.sh
├── requirements.txt
├── scripts/
│   ├── verify_glm_capabilities.py
│   ├── verify_multimodal_upstream.py
│   ├── verify_multimodal_proxy.py
│   └── verify_responses_proxy.py
└── src/
    └── glm_responses_proxy/
        ├── __init__.py
        ├── __main__.py
        ├── request_capture.py
        ├── server.py
        └── traffic_stats.py
```

## Acknowledgments

This project was made possible with the help of:

- **[Codex](https://github.com/openai/codex)** — The codebase was developed with the assistance of OpenAI Codex, which helped write, review, and iterate on the proxy logic, test scripts, and documentation.
- **[GLM](https://github.com/THUDM/GLM-4)** — The proxy is designed to work with GLM series models (GLM-5.1) served via vLLM, and the protocol compatibility was validated against GLM's chat completions API.

## License

MIT

---

# 中文文档

一个轻量级 FastAPI 代理服务，将 OpenAI 风格的 `/v1/responses` 请求翻译为 GLM 兼容上游（如 vLLM）的 `/v1/chat/completions` 请求。

适用于本地或内网部署场景：上游服务已提供 OpenAI 兼容的 Chat API，但客户端需要使用较新的 Responses API。

Python 要求：3.8+

## 功能特性

- 代理 `/v1/responses`、`/v1/chat/completions` 和 `/v1/models` 三个端点
- 将 Responses 请求输入转换为 chat-completions 消息格式
- 将 chat-completions 响应转换回 Responses 输出格式
- 支持流式响应，正确发送 SSE 事件
- 将推理（reasoning）输出映射为 Responses 事件
- 将函数工具、工具调用输出和工具调用流式事件进行映射
- 自动路由：纯文本请求 → 文本上游，图片请求 → 多模态上游
- 内置流量统计（请求数、token 用量、数据传输量）
- 包含调试日志和验证脚本

## 安装

### 方式一：从目录安装

```bash
cd glm-responses-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```

### 方式二：仅安装依赖

```bash
cd glm-responses-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

使用内置默认值直接启动：

```bash
glm-responses-proxy
```

或直接运行模块：

```bash
python -m glm_responses_proxy
```

指定上游地址和监听端口：

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

使用 `run.sh` 后台启动（无需先 `pip install .`）：

```bash
./run.sh start
```

可选环境变量：

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=glm-5.1-fp8 \
MULTIMODAL_BASE_URL=http://localhost:33338/v1 \
MULTIMODAL_MODEL=Qwen/Qwen3-VL-8B-Instruct \
PORT=8080 DEBUG=1 ./run.sh
```

服务管理命令：

```bash
./run.sh start          # 启动
./run.sh start --capture  # 启动并开启请求捕获日志
./run.sh stop           # 停止
./run.sh restart        # 重启
./run.sh restart --capture  # 重启并开启请求捕获日志
./run.sh status         # 查看状态
./run.sh logs           # 查看日志
./run.sh --help         # 帮助
```

## 命令行参数

```text
--base-url             上游 `/v1` 基础地址
--model                文本模型名称
--multimodal-base-url  多模态上游 `/v1` 基础地址
--multimodal-model     多模态模型名称
--host                 监听地址
--port                 监听端口
--log-level            日志级别，如 INFO 或 DEBUG
--debug                开启详细调试日志
```

## 端点

- `GET /` — 服务信息和流量统计
- `GET /v1/models` — 列出可用模型
- `POST /v1/chat/completions` — Chat Completions 代理
- `POST /v1/responses` — Responses API 代理

## 流量统计

代理自动跟踪流量指标，每 60 秒输出一次统计日志。也可以通过 `GET /` 端点查询实时统计。

跟踪指标：

- 各端点请求总数（`/v1/responses`、`/v1/chat/completions`、`/v1/models`）
- 成功和错误计数
- Token 用量（输入 token、输出 token、总 token）
- 数据传输量（发送到上游的字节数、从上游接收的字节数）

日志输出示例：

```
2026-05-16 15:00:00 INFO glm_proxy.traffic [traffic stats] requests=42 responses_stream=28 responses_nonstream=8 chat_stream=4 chat_nonstream=2 models=0 | errors=1 | input_tokens=12340 output_tokens=5678 total_tokens=18018 | bytes_up=256000 bytes_down=1024000 | uptime=3600s
```

## 验证

`scripts/` 目录包含验证脚本：

- `scripts/verify_glm_capabilities.py` — 验证上游 GLM 兼容的 `/v1/chat/completions` 服务
- `scripts/verify_responses_proxy.py` — 验证本地 `/v1/responses` 代理行为
- `scripts/verify_multimodal_upstream.py` — 验证多模态 vLLM 上游的图片输入
- `scripts/verify_multimodal_proxy.py` — 验证多模态 `/v1/responses` 代理行为

示例：

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

## GLM 覆盖范围

当前代理已在 `GLM-5.1-FP8` 上游 `/v1/chat/completions` API 上完成验证。

### 已验证的上游 GLM 能力

- 基础对话补全
- 非流式和流式推理输出
- 函数调用
- 工具调用后续对话（`role="tool"`）
- JSON 模式（`response_format`）
- 流式工具调用参数增量

### 已验证的 Responses 代理行为

- `POST /v1/responses` 基础文本请求
- `POST /v1/responses` 图片输入（`input_image`）
- 流式 Responses 事件（`response.created`、`response.in_progress`、`response.completed`、`data: [DONE]`）
- `reasoning` 映射为 Responses 输出项和流式增量事件
- `function_call` 映射为 Responses 输出项
- `function_call_output` 映射回 chat `tool` 消息用于后续对话
- Codex 风格的 `function` 工具映射为 chat-completions 工具格式
- Codex 风格的 `custom` 工具降级为普通函数工具
- 自动路由：纯文本请求 → 文本上游，图片请求 → 多模态上游

### 重要限制

- 这不是完整的 OpenAI Responses 兼容实现
- OpenAI 托管工具（`web_search`、`file_search`、`computer_use`、`code_interpreter`）通过客户端编排工作：模型决定何时调用，客户端执行调用，结果通过 `function_call_output` 返回。GLM 能正确输出工具调用格式，因此这些工具在客户端支持时可以正常工作。本代理原样传递工具调用，不做修改
- `custom` 工具目前降级为普通函数工具，有助于兼容性但不保留自由格式语义
- 响应持久化功能（`previous_response_id`、存储的响应项、服务端会话状态）未实现
- 多模态支持目前仅限于图片输入路由和图片理解输出
- 通用 `input_file` 解析、PDF 解析和音频输入未完全实现
- 事件形状接近 Codex 风格的 Responses 语义，但不保证在每个字段上都匹配 OpenAI 行为

## 项目结构

```text
glm-responses-proxy/
├── pyproject.toml
├── README.md
├── run.sh
├── requirements.txt
├── scripts/
│   ├── verify_glm_capabilities.py
│   ├── verify_multimodal_upstream.py
│   ├── verify_multimodal_proxy.py
│   └── verify_responses_proxy.py
└── src/
    └── glm_responses_proxy/
        ├── __init__.py
        ├── __main__.py
        ├── request_capture.py
        ├── server.py
        └── traffic_stats.py
```

## 致谢

本项目的完成离不开以下项目的帮助：

- **[Codex](https://github.com/openai/codex)** — 代码库的开发全程使用了 OpenAI Codex 辅助，包括代理逻辑、测试脚本和文档的编写、审查与迭代。
- **[GLM](https://github.com/THUDM/GLM-4)** — 本代理专为对接通过 vLLM 部署的 GLM 系列模型（GLM-5.1）而设计，协议兼容性已针对 GLM 的 Chat Completions API 进行了验证。

## 许可证

MIT
