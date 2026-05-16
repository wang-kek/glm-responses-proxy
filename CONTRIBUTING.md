# Contributing

Thanks for your interest in improving `glm-responses-proxy`!

## Development

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run the verification scripts in `scripts/` to confirm nothing is broken
5. Commit with a clear message
6. Push and open a Pull Request

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
glm-responses-proxy --debug
```

## Verification

```bash
python scripts/verify_glm_capabilities.py --base-url http://localhost:8000/v1 --api-key YOUR_KEY
python scripts/verify_responses_proxy.py --base-url http://localhost:8080 --api-key YOUR_KEY
```

## Code style

- Keep the single-file `server.py` structure for now; split only when readability demands it
- Prefer standard library where possible; use `httpx`/`fastapi`/`uvicorn` for async HTTP and serving
- Write verification scripts in `scripts/` using only `urllib` to avoid extra dependencies

## Reporting issues

Open a GitHub Issue with:

- Proxy version (`glm-responses-proxy --help` or `git rev-parse --short HEAD`)
- Upstream model and serving framework (e.g. vLLM 0.x + GLM-5.1-FP8)
- Minimal request payload that reproduces the problem
- Relevant log output (redact API keys)
