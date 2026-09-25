#!/usr/bin/env bash
# Mixed-layer / sea-ice closure A/B on the 0.5-degree ice-floor candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDA="${LAMBDA:-80}"
DAYS="${DAYS:-365}"
MIXED_LAYER_DEPTH="${MIXED_LAYER_DEPTH:-50}"
ICE_SALT_FLUX="${ICE_SALT_FLUX:-1e-7}"
TAG="${1:-mixed_layer_ice_050_${DAYS}d}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python" ]]; then
  PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
fi
if [[ -d "/mnt/c/Users/zhen.luo/ocean_solver/data/woa" && -z "${OCEAN_SOLVER_WOA_DIR:-}" ]]; then
  export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
fi
export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
OUT_DIR="${OUT_DIR:-results/mixed_layer_ice_050}"
LOG_DIR="${LOG_DIR:-logs/mixed_layer_ice_050}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.5 --resolution-remap area \
  --seasonal-wind --wind-year 2023 --real-air-temp --ice-air-floor \
  --lambda-bulk "${LAMBDA}" \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --nu-h 2e6 --dt 1800 \
  --days "${DAYS}" --snap-days "${DAYS}" \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --mixed-layer-depth "${MIXED_LAYER_DEPTH}" \
  --ice-salt-flux "${ICE_SALT_FLUX}"
