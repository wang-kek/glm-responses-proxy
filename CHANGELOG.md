# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-16

### Added

- Initial release of `glm-responses-proxy`.
- Proxy `/v1/responses` requests to OpenAI-compatible `/v1/chat/completions` backends.
- Proxy `/v1/chat/completions` and `/v1/models` endpoints.
- Streaming and non-streaming response conversion.
- Reasoning output mapped to Responses API `reasoning` items and stream events.
- Function calling and tool-call streaming support.
- Multimodal (image) input routing to a separate vision-model upstream.
- Codex-style `custom` tool downgrade to regular function tools.
- Request capture log for debugging client behavior.
- Verification scripts for upstream capabilities and proxy behavior.
- `run.sh` process manager for local deployment.
