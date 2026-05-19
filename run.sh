#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR="$SCRIPT_DIR/.run"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
PID_FILE="$PID_DIR/proxy.pid"
LOG_FILE="$LOG_DIR/proxy.log"
CAPTURE_LOG_FILE="$SCRIPT_DIR/testhi.log"

mkdir -p "$LOG_DIR" "$PID_DIR"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-glm-5.1-fp8}"
MULTIMODAL_BASE_URL="${MULTIMODAL_BASE_URL:-http://localhost:33338/v1}"
MULTIMODAL_MODEL="${MULTIMODAL_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DEBUG="${DEBUG:-0}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
DOCX_AUTO_DISCOVERY_ENABLED="${DOCX_AUTO_DISCOVERY_ENABLED:-0}"

usage() {
  cat <<EOF
Usage:
  ./run.sh [start|stop|restart|status|logs] [--capture]

Defaults:
  BASE_URL=$BASE_URL
  MODEL=$MODEL
  MULTIMODAL_BASE_URL=$MULTIMODAL_BASE_URL
  MULTIMODAL_MODEL=$MULTIMODAL_MODEL
  HOST=$HOST
  PORT=$PORT
  TOKENIZER_PATH=$TOKENIZER_PATH
  DOCX_AUTO_DISCOVERY_ENABLED=$DOCX_AUTO_DISCOVERY_ENABLED
  CAPTURE_LOG_FILE=$CAPTURE_LOG_FILE
EOF
}

is_running() {
  [ -f "$PID_FILE" ] || return 1
  kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null
}

start() {
  local capture_flag="${1:-}"
  local capture_log=""
  [ "$capture_flag" = "--capture" ] && capture_log="$CAPTURE_LOG_FILE"

  if is_running; then
    echo "proxy already running: pid=$(cat "$PID_FILE")"
    if [ "$capture_flag" = "--capture" ]; then
      echo "capture was not applied to the running process; use: ./run.sh restart --capture"
    fi
    return
  fi

  local cmd=(
    "$PYTHON_BIN" -m glm_responses_proxy
    --base-url "$BASE_URL"
    --model "$MODEL"
    --multimodal-base-url "$MULTIMODAL_BASE_URL"
    --multimodal-model "$MULTIMODAL_MODEL"
    --host "$HOST"
    --port "$PORT"
    --log-level "$LOG_LEVEL"
  )
  [ -n "$TOKENIZER_PATH" ] && cmd+=(--tokenizer "$TOKENIZER_PATH")
  export DOCX_AUTO_DISCOVERY_ENABLED
  [ -n "$capture_log" ] && cmd+=(--capture-log "$capture_log")
  [ "$DEBUG" = "1" ] && cmd+=(--debug)

  nohup "${cmd[@]}" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "started proxy: pid=$(cat "$PID_FILE") log=$LOG_FILE request_capture_log=${capture_log:-<disabled>}"
}

stop() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "proxy is not running"
    return
  fi

  local pid
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "stopped proxy: pid=$pid"
}

status() {
  if is_running; then
    echo "proxy is running: pid=$(cat "$PID_FILE")"
  else
    rm -f "$PID_FILE"
    echo "proxy is not running"
  fi
  echo "log: $LOG_FILE"
}

case "${1:-start}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) stop; start "${2:-}" ;;
  status) status ;;
  logs) touch "$LOG_FILE"; tail -f "$LOG_FILE" ;;
  help|--help|-h) usage ;;
  *) usage; exit 1 ;;
esac
