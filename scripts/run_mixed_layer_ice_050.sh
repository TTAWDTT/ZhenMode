#!/usr/bin/env bash
# Mixed-layer / sea-ice closure A/B on the 0.5-degree ice-floor candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDA="${LAMBDA:-80}"
DAYS="${DAYS:-365}"
MIXED_LAYER_DEPTH="${MIXED_LAYER_DEPTH:-50}"
ICE_SALT_FLUX="${ICE_SALT_FLUX:-1e-7}"
MIXED_LAYER_LAT_BAND="${MIXED_LAYER_LAT_BAND:-}"
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
MIXED_LAYER_ARGS=(--mixed-layer-depth "${MIXED_LAYER_DEPTH}")
if [[ -n "${MIXED_LAYER_LAT_BAND}" ]]; then
  read -r ML_LAT_MIN ML_LAT_MAX <<< "${MIXED_LAYER_LAT_BAND}"
  MIXED_LAYER_ARGS+=(--mixed-layer-lat-band "${ML_LAT_MIN}" "${ML_LAT_MAX}")
elif [[ "${MIXED_LAYER_MODE:-constant}" == "stratification" ]]; then
  MIXED_LAYER_ARGS+=(
    --mixed-layer-mode stratification
    --mld-density-delta "${MLD_DENSITY_DELTA:-0.03}"
    --mixed-layer-depth-min "${MIXED_LAYER_DEPTH_MIN:-10}"
    --mixed-layer-depth-max "${MIXED_LAYER_DEPTH_MAX:-100}"
  )
fi
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
  --days "${DAYS}" --snap-days "${SNAP_DAYS:-10}" \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  "${MIXED_LAYER_ARGS[@]}" \
  --ice-salt-flux "${ICE_SALT_FLUX}" \
  ${DYNAMIC_ICE:+--dynamic-ice} \
  ${DYNAMIC_ICE:+--ice-insulation-scale-m "${ICE_INSULATION_SCALE_M:-1}"}

