"""Traffic statistics for the GLM Responses Proxy.

Tracks request counts, token usage, and data volume per endpoint.
Periodically logs a summary and exposes stats via the /stats endpoint.
"""

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger("glm_proxy.traffic")


class TrafficStats:
    """Thread-safe traffic counter with periodic logging."""

    def __init__(self, log_interval: float = 60.0):
        self._lock = threading.Lock()
        self._log_interval = log_interval
        self._started_at = time.time()

        # Per-endpoint counters
        self._counters: Dict[str, Dict[str, Any]] = {}

        # Periodic logger state
        self._timer: threading.Timer | None = None

    # ---- internal helpers ----

    def _ensure_endpoint(self, endpoint: str):
        if endpoint not in self._counters:
            self._counters[endpoint] = {
                "requests": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "request_bytes": 0,
                "response_bytes": 0,
            }

    def _periodic_log(self):
        self.log_summary()
        self._timer = threading.Timer(self._log_interval, self._periodic_log)
        self._timer.daemon = True
        self._timer.start()

    # ---- public API ----

    def start_periodic_logging(self):
        """Begin logging traffic summaries at the configured interval."""
        self._timer = threading.Timer(self._log_interval, self._periodic_log)
        self._timer.daemon = True
        self._timer.start()

    def stop_periodic_logging(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def record_request(
        self,
        endpoint: str,
        *,
        request_bytes: int = 0,
        response_bytes: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
    ):
        with self._lock:
            self._ensure_endpoint(endpoint)
            c = self._counters[endpoint]
            c["requests"] += 1
            c["request_bytes"] += request_bytes
            c["response_bytes"] += response_bytes
            c["input_tokens"] += input_tokens
            c["output_tokens"] += output_tokens
            if error:
                c["errors"] += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            uptime_s = time.time() - self._started_at
            endpoints = {}
            total_requests = 0
            total_errors = 0
            total_input_tokens = 0
            total_output_tokens = 0
            total_request_bytes = 0
            total_response_bytes = 0

            for name, c in sorted(self._counters.items()):
                endpoints[name] = dict(c)
                total_requests += c["requests"]
                total_errors += c["errors"]
                total_input_tokens += c["input_tokens"]
                total_output_tokens += c["output_tokens"]
                total_request_bytes += c["request_bytes"]
                total_response_bytes += c["response_bytes"]

            return {
                "uptime_s": round(uptime_s, 1),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_request_bytes": total_request_bytes,
                "total_response_bytes": total_response_bytes,
                "endpoints": endpoints,
            }

    def log_summary(self):
        summary = self.get_summary()
        parts = [
            f"uptime={summary['uptime_s']}s",
            f"requests={summary['total_requests']}",
            f"errors={summary['total_errors']}",
            f"in_tokens={summary['total_input_tokens']}",
            f"out_tokens={summary['total_output_tokens']}",
            f"req_bytes={summary['total_request_bytes']}",
            f"resp_bytes={summary['total_response_bytes']}",
        ]
        logger.info("traffic summary: %s", " ".join(parts))

        for name, c in sorted(summary["endpoints"].items()):
            logger.info(
                "  %-25s requests=%d errors=%d in_tok=%d out_tok=%d req_kb=%.1f resp_kb=%.1f",
                name,
                c["requests"],
                c["errors"],
                c["input_tokens"],
                c["output_tokens"],
                c["request_bytes"] / 1024,
                c["response_bytes"] / 1024,
            )


# Global singleton
traffic_stats = TrafficStats()
