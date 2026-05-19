# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-20

### Fixed

- Avoided double-converting Windows file paths that already appear inside Markdown links, preventing malformed targets like `[file](C:[file](/work/...))` that break Codex file-output rendering on Windows.
- Added `inject_markdown_file_links` post-processing to convert bare file paths in model output into clickable Markdown links, enabling Codex Desktop to display file outputs in the right-side panel.
- Replaced `DOCX_NAME_PATTERN` with absolute-path-only matching in DOCX auto-discovery to avoid false-positive relative-name matches.
- Gated DOCX auto-discovery behind `DOCX_AUTO_DISCOVERY_ENABLED` env var (default off) to prevent unnecessary file scanning.
- Added `run.sh` support for `DOCX_AUTO_DISCOVERY_ENABLED` environment variable passthrough.



## [0.1.1] - 2026-05-18

### Added

- Context protection pipeline for upstream chat requests, including token estimation, old-message summarization, soft trimming, and hard rejection for oversized payloads.
- Optional tokenizer-based token estimation with heuristic fallback for local deployments that can provide a tokenizer path.
- DOCX text extraction and prompt injection for document-reading workflows, with automatic nearby-file discovery for common Codex document tasks.
- Test-run sync script `scripts/sync_test_run.sh` for refreshing an isolated validation directory from the main source tree, with default log cleanup and optional log preservation.

### Changed

- Improved Responses-to-Chat tool history mapping so prior `function_call` items are preserved as assistant tool calls instead of dropping call intent.
- Improved handling of aborted `apply_patch` results to discourage repeated identical retries and guide the model toward a revised patch or direct textual output.
- Normalized upstream `response_format` conversion for `json_schema`, `json_object`, and `text` request styles.
- Added `docx_injected` context-preparation logging for both `/v1/responses` and `/v1/chat/completions`.
- Truncated oversized tool outputs before re-injecting them into model context, with summary metadata in logs to prevent large shell/manifest outputs from causing runaway context growth during PPT and template workflows.

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
