#!/usr/bin/env bash
# Reusable Stage-F ocean_solver probe with optional mixed-layer/ice closure.
# Scientific base: monthly 2m air, dynamic bulk lambda, no restoring, no
# prescribed meridional heat flux, 500 m bathymetric floor.
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS="${DAYS:-30}"
LAMBDA="${LAMBDA:-80}"
MIXED_LAYER_DEPTH="${MIXED_LAYER_DEPTH:-}"
MIXED_LAYER_LAT_BAND="${MIXED_LAYER_LAT_BAND:-}"
MIXED_LAYER_MODE="${MIXED_LAYER_MODE:-constant}"
DYNAMIC_ICE="${DYNAMIC_ICE:-false}"
ICE_SALT_FLUX="${ICE_SALT_FLUX:-1e-7}"
ICE_INSULATION_SCALE_M="${ICE_INSULATION_SCALE_M:-1}"
SNAP_DAYS="${SNAP_DAYS:-10}"
SAVE_3D="${SAVE_3D:-false}"
SAVE_3D_TERMS="${SAVE_3D_TERMS:-false}"
TAG="${1:-industrial_comparison_050_stage_f_probe_${DAYS}d}"
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

MIXED_LAYER_ARGS=()
if [[ -n "${MIXED_LAYER_DEPTH}" ]]; then
  MIXED_LAYER_ARGS+=(--mixed-layer-depth "${MIXED_LAYER_DEPTH}")
fi
if [[ -n "${MIXED_LAYER_LAT_BAND}" ]]; then
  read -r ML_LAT_MIN ML_LAT_MAX <<< "${MIXED_LAYER_LAT_BAND}"
  MIXED_LAYER_ARGS+=(--mixed-layer-lat-band "${ML_LAT_MIN}" "${ML_LAT_MAX}")
elif [[ "${MIXED_LAYER_MODE}" == "stratification" ]]; then
  MIXED_LAYER_ARGS+=(
    --mixed-layer-mode stratification
    --mld-density-delta "${MLD_DENSITY_DELTA:-0.03}"
    --mixed-layer-depth-min "${MIXED_LAYER_DEPTH_MIN:-10}"
    --mixed-layer-depth-max "${MIXED_LAYER_DEPTH_MAX:-100}"
  )
fi

DYNAMIC_ICE_ARGS=()
if [[ "${DYNAMIC_ICE}" == "true" ]]; then
  DYNAMIC_ICE_ARGS+=(--dynamic-ice --ice-insulation-scale-m "${ICE_INSULATION_SCALE_M}")
fi

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
  --days "${DAYS}" --snap-days "${SNAP_DAYS}" \
  $( [[ "${SAVE_3D}" == "true" ]] && echo --save-3d ) \
  $( [[ "${SAVE_3D_TERMS}" == "true" ]] && echo --save-3d-terms ) \
  --tag "${TAG}" \
  --out-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  "${MIXED_LAYER_ARGS[@]}" \
  --ice-salt-flux "${ICE_SALT_FLUX}" \
  "${DYNAMIC_ICE_ARGS[@]}"
