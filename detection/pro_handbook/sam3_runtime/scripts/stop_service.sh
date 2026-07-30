#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="${runtime_dir}/logs/service.pid"
test -f "${pid_file}" || { echo "service is not running"; exit 0; }
pid="$(cat "${pid_file}")"
kill "${pid}" 2>/dev/null || true
rm -f "${pid_file}"
echo "stopped PID ${pid}"
