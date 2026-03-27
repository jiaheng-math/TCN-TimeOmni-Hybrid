#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
  install)
    python -m pip install -r requirements.txt
    ;;
  preprocess)
    python scripts/preprocess_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml "$@"
    ;;
  train)
    python scripts/train_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml "$@"
    ;;
  train-point)
    python scripts/train_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml "$@"
    ;;
  eval)
    python scripts/evaluate_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml "$@"
    ;;
  eval-point)
    python scripts/evaluate_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml "$@"
    ;;
  visualize)
    python scripts/visualize_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml "$@"
    ;;
  visualize-point)
    python scripts/visualize_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml "$@"
    ;;
  hybrid)
    python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_local.yaml "$@"
    ;;
  hybrid-point)
    python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_point_local.yaml "$@"
    ;;
  full)
    python scripts/run_full_pipeline.py \
      --tcn-config configs/tcn/fd001_tcn_uncertainty_tuned.yaml \
      --hybrid-config configs/hybrid/fd001_hybrid_local.yaml \
      "$@"
    ;;
  full-point)
    python scripts/run_full_pipeline.py \
      --tcn-config configs/tcn/fd001_tcn_point_tuned.yaml \
      --hybrid-config configs/hybrid/fd001_hybrid_point_local.yaml \
      "$@"
    ;;
  help|*)
    cat <<'EOF'
Usage:
  ./run.sh install
  ./run.sh preprocess
  ./run.sh train
  ./run.sh train-point
  ./run.sh eval
  ./run.sh eval-point
  ./run.sh visualize
  ./run.sh visualize-point
  ./run.sh hybrid
  ./run.sh hybrid-point
  ./run.sh full
  ./run.sh full-point

Environment:
  Copy .env.example and export the variables you need before running.
EOF
    ;;
esac
