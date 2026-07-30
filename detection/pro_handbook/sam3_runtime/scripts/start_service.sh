#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${runtime_dir}/.venv/bin/python"
test -x "${python_bin}" || { echo "Missing ${python_bin}; run create_env.sh first" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="${runtime_dir}/../../.."
unset LD_LIBRARY_PATH
nohup "${python_bin}" -m detection.pro_handbook.sam3_runtime.service.service >>"${runtime_dir}/logs/service.stdout.log" 2>&1 &
echo $! >"${runtime_dir}/logs/service.pid"
echo "started PID $!"
