#!/usr/bin/env bash
# Wrap a runner command with a whole-process timeout.
# Sync-only timeouts can miss hangs during compile, launch, or driver teardown.
set -euo pipefail

usage() {
  echo "usage: $0 [--timeout SECONDS] <command...>" >&2
  exit 2
}

TIMEOUT_SECS="${PTO_PROCESS_TIMEOUT_S:-}"
KILL_AFTER="${PTO_PROCESS_KILL_AFTER_S:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      [[ $# -ge 2 ]] || usage
      TIMEOUT_SECS="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      usage
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -gt 0 ]] || usage

if [[ -z "${TIMEOUT_SECS}" ]]; then
  if [[ "${PTO_SIMULATOR:-}" == "1" ]]; then
    TIMEOUT_SECS=1800
  else
    TIMEOUT_SECS=60
  fi
fi

if command -v timeout >/dev/null 2>&1; then
  echo "[pto] whole-process timeout: ${TIMEOUT_SECS}s (override with PTO_PROCESS_TIMEOUT_S)" >&2
  exec timeout --foreground -k "${KILL_AFTER}" "${TIMEOUT_SECS}" "$@"
fi

echo "[pto] warning: GNU timeout not found; running without whole-process timeout" >&2
exec "$@"
