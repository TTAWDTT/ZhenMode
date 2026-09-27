#!/usr/bin/env bash
# Stage-F ocean_solver side with band-limited dynamic ice only.
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS="${DAYS:-30}"
LAMBDA="${LAMBDA:-80}"
ICE_LAT_MIN="${ICE_LAT_MIN:-40}"
ICE_LAT_MAX="${ICE_LAT_MAX:-65}"
SAVE_3D="${SAVE_3D:-false}"
SAVE_3D_TERMS="${SAVE_3D_TERMS:-false}"
TAG="${1:-global_stage_f_dyn_ice_lat${ICE_LAT_MIN}_${ICE_LAT_MAX}_probe_${DAYS}d}"
OUT_DIR="${OUT_DIR:-results/industrial_comparison_045}"
LOG_DIR="${LOG_DIR:-logs/industrial_comparison_045}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python" ]]; then
  PYTHON_BIN="/mnt/c/Users/zhen.luo/ocean_solver/.venv-gpu-jax/bin/python"
fi
if [[ -d "/mnt/c/Users/zhen.luo/ocean_solver/data/woa" && -z "${OCEAN_SOLVER_WOA_DIR:-}" ]]; then
  export OCEAN_SOLVER_WOA_DIR="/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
fi
export OCEAN_SOLVER_BATHYMETRY="${OCEAN_SOLVER_BATHYMETRY:-/mnt/c/Users/zhen.luo/Desktop/ETOPO_2022_v1_r3600x1800_surface.nc}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" src/run_long_integration_global.py \
  --mode-split --use-scan --dtype float32 \
  --lat-max 65 --resolution 0.5 --resolution-remap area \
  --seasonal-wind --wind-year 2023 \
  --real-air-temp-monthly \
  --no-meridional-heat-flux \
  --lambda-bulk "${LAMBDA}" \
  --kappa-v 1e-6 --kappa-conv 0.01 \
  --kappa-gm 0 \
  --localize-conv --fct-adv --project-adv-vel \
  --min-depth 500 --smooth-passes 80 \
  --nu-h 2e6 --dt 1800 \
  --dynamic-ice --dynamic-ice-lat-band "${ICE_LAT_MIN}" "${ICE_LAT_MAX}" \
  --days "${DAYS}" --snap-days "${SNAP_DAYS:-10}" \
  $( [[ "${SAVE_3D}" == "true" ]] && echo --save-3d ) \
  $( [[ "${SAVE_3D_TERMS}" == "true" ]] && echo --save-3d-terms ) \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}"
