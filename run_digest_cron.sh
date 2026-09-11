#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-$SCRIPT_DIR/ollama-cron.log}"
STARTED_OLLAMA=0
OLLAMA_PID=""

# Cron has a minimal PATH. Include common Intel and Apple Silicon locations.
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ollama_is_ready() {
    curl --silent --show-error --fail --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1
}

find_ollama() {
    if [ -n "${OLLAMA_BIN:-}" ] && [ -x "$OLLAMA_BIN" ]; then
        printf '%s\n' "$OLLAMA_BIN"
        return 0
    fi

    local detected
    detected="$(command -v ollama 2>/dev/null || true)"
    if [ -n "$detected" ]; then
        printf '%s\n' "$detected"
        return 0
    fi

    for detected in \
        /usr/local/bin/ollama \
        /opt/homebrew/bin/ollama \
        /Applications/Ollama.app/Contents/Resources/ollama
    do
        if [ -x "$detected" ]; then
            printf '%s\n' "$detected"
            return 0
        fi
    done
    return 1
}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ "$STARTED_OLLAMA" -eq 1 ] && [ -n "$OLLAMA_PID" ]; then
        log "Stopping the Ollama server started by this job (PID $OLLAMA_PID)."
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    exit "$status"
}

trap cleanup EXIT INT TERM

if [ ! -x "$PYTHON_BIN" ]; then
    log "ERROR: Virtual-environment Python not found at $PYTHON_BIN"
    log "Run: python3 -m venv '$SCRIPT_DIR/.venv' && '$SCRIPT_DIR/.venv/bin/pip' install -r '$SCRIPT_DIR/requirements.txt'"
    exit 1
fi

if ollama_is_ready; then
    log "Ollama is already running; it will be left running after this job."
else
    if ! OLLAMA_EXECUTABLE="$(find_ollama)"; then
        log "ERROR: Ollama executable not found. Set OLLAMA_BIN to its absolute path."
        exit 1
    fi

    log "Starting Ollama with $OLLAMA_EXECUTABLE"
    "$OLLAMA_EXECUTABLE" serve >>"$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!
    STARTED_OLLAMA=1

    attempt=0
    while ! ollama_is_ready; do
        attempt=$((attempt + 1))
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            log "ERROR: Ollama exited before becoming ready. See $OLLAMA_LOG"
            exit 1
        fi
        if [ "$attempt" -ge 30 ]; then
            log "ERROR: Ollama did not become ready within 30 seconds. See $OLLAMA_LOG"
            exit 1
        fi
        sleep 1
    done
    log "Ollama is ready."
fi

log "Generating the AI research digest."
cd "$SCRIPT_DIR"
"$PYTHON_BIN" main.py "$@"
log "Digest generation completed successfully."
