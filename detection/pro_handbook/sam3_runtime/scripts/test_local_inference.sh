#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1 || { echo "usage: $0 IMAGE" >&2; exit 2; }
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${runtime_dir}/../../.."
unset LD_LIBRARY_PATH
"${runtime_dir}/.venv/bin/python" -m detection.pro_handbook.sam3_runtime.scripts.test_client "$1"
