#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${runtime_dir}/.venv/bin/python"
test -x "${python_bin}" || { echo "Missing ${python_bin}; run create_env.sh first" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="${runtime_dir}/../../.."
# torch 2.10.0+cu128 must not resolve CUDA 12.2 cuBLAS from the ROS process
# environment. Preserve every unrelated ROS/xArm path instead of unsetting the
# complete LD_LIBRARY_PATH.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  filtered_ld_library_path=""
  IFS=':' read -r -a ld_library_parts <<<"${LD_LIBRARY_PATH}"
  for part in "${ld_library_parts[@]}"; do
    [[ "${part}" == "/usr/local/cuda-12.2/lib64" ]] && continue
    if [[ -z "${filtered_ld_library_path}" ]]; then
      filtered_ld_library_path="${part}"
    else
      filtered_ld_library_path="${filtered_ld_library_path}:${part}"
    fi
  done
  if [[ -n "${filtered_ld_library_path}" ]]; then
    export LD_LIBRARY_PATH="${filtered_ld_library_path}"
  else
    unset LD_LIBRARY_PATH
  fi
fi
nohup "${python_bin}" -m detection.pro_handbook.sam3_runtime.service.service >>"${runtime_dir}/logs/service.stdout.log" 2>&1 &
echo $! >"${runtime_dir}/logs/service.pid"
echo "started PID $!"
