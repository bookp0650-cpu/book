#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
micromamba="${runtime_dir}/vendor/bin/micromamba"
test -x "${micromamba}" || { echo "Missing ${micromamba}" >&2; exit 1; }
test ! -e "${runtime_dir}/.venv" || { echo "Refusing to overwrite existing ${runtime_dir}/.venv" >&2; exit 1; }
MAMBA_ROOT_PREFIX="${runtime_dir}/.micromamba" "${micromamba}" create -y -p "${runtime_dir}/.venv" python=3.12.13 pip=26.1.1 setuptools=80.10.2
echo "Install PyTorch cu128, configs/requirements-inference-direct.txt, then editable sam3_source as documented."
