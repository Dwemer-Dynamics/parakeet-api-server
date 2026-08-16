#!/bin/bash

# Parakeet STT API Server - Startup Script (Linux/macOS)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STARTUP_TIMEOUT_SECONDS="${PARAKEET_STARTUP_TIMEOUT_SECONDS:-900}"
STARTUP_ATTEMPTS="${PARAKEET_STARTUP_ATTEMPTS:-2}"
STARTUP_POLL_SECONDS="${PARAKEET_STARTUP_POLL_SECONDS:-2}"
SERVER_PORT=8022

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
    esac
done

for value_name in STARTUP_TIMEOUT_SECONDS STARTUP_ATTEMPTS STARTUP_POLL_SECONDS SERVER_PORT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer, got: $value" >&2
        exit 2
    fi
done

HEALTH_URL="http://127.0.0.1:${SERVER_PORT}/health"
PID_FILE="${TMPDIR:-/tmp}/parakeet-api-server-${UID}-${SERVER_PORT}.pid"
active_pid=""

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

handle_signal() {
    echo "Startup supervisor interrupted; stopping Parakeet"
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

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required for the Parakeet startup health check." >&2
    exit 1
fi

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

last_status=1
for ((attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++)); do
    echo "Starting server (attempt $attempt/$STARTUP_ATTEMPTS)..."
    echo "Startup deadline: ${STARTUP_TIMEOUT_SECONDS}s; health: $HEALTH_URL"
    echo ""

    PYTHONUNBUFFERED=1 python server.py "$@" &
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
            echo "The model load is stalled; terminating this attempt before retrying." >&2
            stop_active_server
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
        active_pid=""
        rm -f "$PID_FILE"
    fi

    if (( attempt < STARTUP_ATTEMPTS )); then
        echo "Retrying Parakeet startup in 2 seconds..."
        sleep 2
    fi
done

echo "ERROR: Parakeet failed to become healthy after $STARTUP_ATTEMPTS attempts." >&2
exit "$last_status"
