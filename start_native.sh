#!/bin/bash

# Parakeet STT API Server - Startup Script (Linux/macOS)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PREPARE_TIMEOUT_SECONDS="${PARAKEET_MODEL_PREPARE_TIMEOUT_SECONDS:-1800}"
STARTUP_TIMEOUT_SECONDS="${PARAKEET_STARTUP_TIMEOUT_SECONDS:-600}"
STARTUP_ATTEMPTS="${PARAKEET_STARTUP_ATTEMPTS:-2}"
STARTUP_POLL_SECONDS="${PARAKEET_STARTUP_POLL_SECONDS:-2}"
SERVER_PORT=8022
SERVER_PRECISION=fp32
FORCE_CPU=false

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
    case "${args[$index]}" in
        --port)
            if (( index + 1 < ${#args[@]} )); then
                SERVER_PORT="${args[$((index + 1))]}"
            fi
            ;;
        --port=*)
            SERVER_PORT="${args[$index]#--port=}"
            ;;
        --precision|-p)
            if (( index + 1 < ${#args[@]} )); then
                SERVER_PRECISION="${args[$((index + 1))]}"
            fi
            ;;
        --precision=*)
            SERVER_PRECISION="${args[$index]#--precision=}"
            ;;
        --cpu)
            FORCE_CPU=true
            ;;
    esac
done

for value_name in MODEL_PREPARE_TIMEOUT_SECONDS STARTUP_TIMEOUT_SECONDS STARTUP_ATTEMPTS STARTUP_POLL_SECONDS SERVER_PORT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer, got: $value" >&2
        exit 2
    fi
done

HEALTH_URL="http://127.0.0.1:${SERVER_PORT}/health"
PID_FILE="${TMPDIR:-/tmp}/parakeet-api-server-${UID}-${SERVER_PORT}.pid"
active_pid=""
PYTHON_BIN=python3

set_startup_state() {
    "$PYTHON_BIN" startup_state.py set "$@" || true
}

# Stop the active child cleanly, escalating only when it ignores SIGTERM.
stop_active_server() {
    local pid="${active_pid:-}"
    local wait_attempt

    if [ -z "$pid" ]; then
        return
    fi

    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for wait_attempt in {1..5}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "Startup process $pid ignored SIGTERM; sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi

    wait "$pid" 2>/dev/null || true
    active_pid=""
    rm -f "$PID_FILE"
}

# Require the Parakeet health response, not merely an occupied TCP port.
server_is_healthy() {
    curl --fail --silent --show-error --max-time 2 "$HEALTH_URL" 2>/dev/null \
        | grep -q '"status":"healthy"'
}

# Capture enough local state to distinguish CUDA, process, memory, and kernel failures.
capture_failure_snapshot() {
    local reason="$1"
    local process_id="${2:-}"

    echo ""
    echo "--- Parakeet failure snapshot: $reason ---"
    "$PYTHON_BIN" startup_state.py show 2>/dev/null || true
    if [ -n "$process_id" ]; then
        ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$process_id" 2>/dev/null || true
    fi
    free -h 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        timeout 10s nvidia-smi \
            --query-gpu=driver_version,name,compute_cap,memory.total,memory.used,memory.free \
            --format=csv,noheader 2>/dev/null || true
    fi
    dmesg 2>/dev/null | grep -Ei 'out of memory|killed process|oom-kill|NVRM|Xid' | tail -n 20 || true
    echo "--- End Parakeet failure snapshot ---"
    echo ""
}

handle_signal() {
    echo "Startup supervisor interrupted; stopping Parakeet"
    set_startup_state "interrupted" "Startup supervisor received a termination signal"
    stop_active_server
    exit 130
}

trap handle_signal INT TERM

echo "========================================"
echo "Parakeet STT API Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run the installation script first:"
    echo "  ./install.sh"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
PYTHON_BIN=python

for required_command in curl timeout; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: $required_command is required for supervised Parakeet startup." >&2
        set_startup_state "startup_failed" "Required command is unavailable" --detail "command=$required_command"
        exit 1
    fi
done

if server_is_healthy; then
    echo "Parakeet is already healthy at $HEALTH_URL"
    exit 0
fi

if command -v flock >/dev/null 2>&1; then
    lock_file="${TMPDIR:-/tmp}/parakeet-api-server-${UID}-${SERVER_PORT}.lock"
    exec 9>"$lock_file"
    if ! flock -n 9; then
        echo "Another Parakeet startup is already in progress for port $SERVER_PORT"
        exit 0
    fi
fi

runtime_args=(check-runtime)
if [ "$SERVER_PRECISION" = "fp32" ] && [ "$FORCE_CPU" = false ]; then
    runtime_args+=(--require-cuda)
fi

if "$PYTHON_BIN" startup_state.py "${runtime_args[@]}"; then
    :
else
    runtime_status=$?
    capture_failure_snapshot "runtime compatibility check failed"
    exit "$runtime_status"
fi

echo "Preparing $SERVER_PRECISION model cache (deadline: ${MODEL_PREPARE_TIMEOUT_SECONDS}s)..."
if timeout --signal=TERM "$MODEL_PREPARE_TIMEOUT_SECONDS" \
    "$PYTHON_BIN" -u model_downloader.py --precision "$SERVER_PRECISION"; then
    :
else
    prepare_status=$?
    set_startup_state \
        "model_prepare_failed" \
        "Model cache preparation failed or timed out" \
        --detail "precision=$SERVER_PRECISION" \
        --detail "status=$prepare_status"
    capture_failure_snapshot "model cache preparation failed"
    exit "$prepare_status"
fi

last_status=1
for ((attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++)); do
    echo "Starting server (attempt $attempt/$STARTUP_ATTEMPTS)..."
    echo "Startup deadline: ${STARTUP_TIMEOUT_SECONDS}s; health: $HEALTH_URL"
    echo ""

    PARAKEET_STARTUP_ATTEMPT="$attempt" "$PYTHON_BIN" startup_state.py set \
        "starting_attempt" \
        "Starting Parakeet server process" \
        --detail "attempt=$attempt" \
        --detail "precision=$SERVER_PRECISION"
    env \
        PARAKEET_STARTUP_ATTEMPT="$attempt" \
        PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" server.py "$@" &
    active_pid=$!
    printf '%s\n' "$active_pid" > "$PID_FILE"
    started_at=$SECONDS

    while kill -0 "$active_pid" 2>/dev/null; do
        if server_is_healthy; then
            elapsed=$((SECONDS - started_at))
            echo "Parakeet health check passed after ${elapsed}s (PID $active_pid)"
            if wait "$active_pid"; then
                last_status=0
            else
                last_status=$?
            fi
            active_pid=""
            rm -f "$PID_FILE"
            exit "$last_status"
        fi

        elapsed=$((SECONDS - started_at))
        if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
            echo "ERROR: Parakeet did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s (PID $active_pid)." >&2
            echo "Startup exceeded its deadline; terminating this attempt before retrying." >&2
            capture_failure_snapshot "startup deadline exceeded" "$active_pid"
            stop_active_server
            set_startup_state \
                "startup_timed_out" \
                "Parakeet did not become healthy before the startup deadline" \
                --detail "attempt=$attempt" \
                --detail "timeout_seconds=$STARTUP_TIMEOUT_SECONDS"
            last_status=124
            break
        fi

        sleep "$STARTUP_POLL_SECONDS"
    done

    if [ -n "$active_pid" ]; then
        if wait "$active_pid"; then
            last_status=0
        else
            last_status=$?
        fi
        echo "ERROR: Parakeet exited before becoming healthy (status $last_status)." >&2
        capture_failure_snapshot "server process exited before health" "$active_pid"
        active_pid=""
        rm -f "$PID_FILE"
        set_startup_state \
            "process_exited" \
            "Parakeet exited before becoming healthy" \
            --detail "attempt=$attempt" \
            --detail "status=$last_status"
    fi

    if (( attempt < STARTUP_ATTEMPTS )); then
        echo "Retrying Parakeet startup in 2 seconds..."
        sleep 2
    fi
done

echo "ERROR: Parakeet failed to become healthy after $STARTUP_ATTEMPTS attempts." >&2
set_startup_state \
    "startup_failed" \
    "Parakeet exhausted all supervised startup attempts" \
    --detail "attempts=$STARTUP_ATTEMPTS" \
    --detail "status=$last_status"
exit "$last_status"
