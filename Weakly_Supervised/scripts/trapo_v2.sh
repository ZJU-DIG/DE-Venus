#!/usr/bin/env bash
# TraPO v2 unified launcher: pipeline | loop | train | promote | probe | final
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=trapo_v2_lib.sh
source "$SCRIPT_DIR/trapo_v2_lib.sh"

trapo_v2_cmd_loop() {
  TRAIN_PARQUET=$BASE_PARQUET
  BEST_VAL=""
  VAL_STALL=0
  STATE_FILE=$ITER_ROOT/loop_state.json
  FINAL_PARQUET=$ITER_ROOT/final_train.parquet
  mkdir -p "$ITER_ROOT"

  echo "=== TraPO v2 iterative loop (val average early stop) ==="
  echo "BASE_MODEL=$BASE_MODEL"
  echo "BASE_PARQUET=$BASE_PARQUET"
  echo "ITER_ROOT=$ITER_ROOT"
  echo "PASSRATE_MIN=$PASSRATE_MIN  SLOPE_MIN=$SLOPE_MIN"

  export TRAPO_WARMUP_STEPS_BASE=${TRAPO_WARMUP_STEPS_BASE:-$TRAPO_WARMUP_STEPS}
  export TRAPO_PROBE_INTERVAL_BASE=${TRAPO_PROBE_INTERVAL_BASE:-$TRAPO_PROBE_INTERVAL}
  N_LABELED_PREV=

  for ITER in $(seq 0 "$MAX_ITER"); do
    echo "---------- iteration $ITER ----------"
    COUNTS=$(python3 -c "
from Weakly_Supervised.utils.analysis.build_stage2_dataset import count_labeled_unlabeled_rows
import json
print(json.dumps(count_labeled_unlabeled_rows('${TRAIN_PARQUET}')))
")
    N_UNLABELED=$(echo "$COUNTS" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_unlabeled'])")
    N_LABELED=$(echo "$COUNTS" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_labeled'])")
    echo "train_parquet=$TRAIN_PARQUET  counts=$COUNTS"
    trapo_v2_apply_scaled_probe_schedule "$N_LABELED"

    if [[ "$N_UNLABELED" -eq 0 ]]; then
      echo "All samples are labeled. Done."
      python3 -c "import shutil; shutil.copy2('${TRAIN_PARQUET}', '${FINAL_PARQUET}')"
      break
    fi

    export ITER TRAIN_PARQUET BASE_MODEL ITER_ROOT ROOT SAVE_ROOT MODEL_NAME EXP_NAME
    unset CKPT_DIR LOG_DIR TRAPO_CKPT_DIR TRAPO_LOG_DIR
    LOG_DIR=$ITER_ROOT/iter_${ITER}/log
    PROBE_DIR=$LOG_DIR/probe_dumps
    set +e
    set -x
    trapo_v2_run_train "$@"
    TRAIN_EXIT=$?
    set -e
    set +x
    if [[ "$TRAIN_EXIT" -ne 0 ]]; then
      if [[ -f "$PROBE_DIR/all_probes_summary.jsonl" ]]; then
        echo "WARN: training exited $TRAIN_EXIT but probe dump exists; continuing promote."
      else
        echo "ERROR: training failed ($TRAIN_EXIT) and no $PROBE_DIR/all_probes_summary.jsonl"
        exit "$TRAIN_EXIT"
      fi
    fi
    if [[ ! -f "$PROBE_DIR/all_probes_summary.jsonl" ]]; then
      echo "ERROR: missing $PROBE_DIR/all_probes_summary.jsonl"
      exit 1
    fi

    VAL_JSON=$ITER_ROOT/iter_${ITER}/val_summary.json
    python3 -c "
from Weakly_Supervised.utils.training_utils import read_latest_val_average
import json, pathlib
info = read_latest_val_average('${LOG_DIR}')
pathlib.Path('${VAL_JSON}').write_text(json.dumps(info if info else {}, indent=2))
"
    VAL_AVG=$(python3 -c "import json; d=json.load(open('${VAL_JSON}')); print(d.get('average',''))")
    VAL_STEP=$(python3 -c "import json; d=json.load(open('${VAL_JSON}')); print(d.get('global_step',''))")

    NEXT_PARQUET=$ITER_ROOT/iter_${ITER}/train_next.parquet
    trapo_v2_promote "$TRAIN_PARQUET" "$PROBE_DIR" "$NEXT_PARQUET" "$ITER"

    META=$ITER_ROOT/iter_${ITER}/train_next.advance_meta.json
    N_NEW=$(python3 -c "import json; print(json.load(open('$META'))['n_newly_promoted'])")
    N_REM=$(python3 -c "import json; print(json.load(open('$META'))['n_unlabeled'])")

    echo "iteration=$ITER  val_average=$VAL_AVG  val_step=$VAL_STEP  new_promoted=$N_NEW  remaining_unlabeled=$N_REM"

    python3 <<PY
import json, pathlib
meta = json.loads(pathlib.Path("$META").read_text())
val_info = json.loads(pathlib.Path("$VAL_JSON").read_text())
state = {
    "iteration": $ITER,
    "train_parquet_in": "$TRAIN_PARQUET",
    "train_parquet_out": "$NEXT_PARQUET",
    "probe_dir": "$PROBE_DIR",
    "log_dir": "$LOG_DIR",
    "n_unlabeled_before": $N_UNLABELED,
    "n_newly_promoted": int("$N_NEW"),
    "n_unlabeled_after": int("$N_REM"),
    "val_average": val_info.get("average"),
    "val_global_step": val_info.get("global_step"),
    "val_per_source": val_info.get("per_source"),
    "promotion_summary": meta.get("promotion_summary"),
}
path = pathlib.Path("$STATE_FILE")
history = json.loads(path.read_text()).get("history", []) if path.exists() else []
history.append(state)
path.write_text(json.dumps({"history": history, "last": state}, indent=2))
PY

    cp -f "$NEXT_PARQUET" "$FINAL_PARQUET"

    if [[ "$N_NEW" -eq 0 ]]; then
      echo "No new promotions; stopping."
      break
    fi

    if [[ -z "$VAL_AVG" ]]; then
      echo "WARN: no val average in $LOG_DIR/valid/; skip val early-stop check this round."
    elif [[ -n "$BEST_VAL" ]]; then
      IMPROVED=$(python3 -c "print(1 if float('$VAL_AVG') > float('$BEST_VAL') + float('$VAL_MIN_DELTA') else 0)")
      if [[ "$IMPROVED" -eq 1 ]]; then
        BEST_VAL=$VAL_AVG
        VAL_STALL=0
        echo "val average improved -> $VAL_AVG"
      else
        VAL_STALL=$((VAL_STALL + 1))
        echo "val average plateau: $VAL_AVG vs best=$BEST_VAL  stall=$VAL_STALL / patience=$VAL_PATIENCE"
        if [[ "$VAL_STALL" -ge "$VAL_PATIENCE" ]]; then
          echo "Early stop: validation average not improving."
          break
        fi
      fi
    else
      BEST_VAL=$VAL_AVG
      echo "initial val average=$BEST_VAL"
    fi

    TRAIN_PARQUET=$NEXT_PARQUET
  done

  echo "Loop finished. final_train_parquet=$FINAL_PARQUET"
  echo "State: $STATE_FILE"
}

trapo_v2_cmd_pipeline() {
  FINAL_PARQUET=${FINAL_PARQUET:-$ITER_ROOT/final_train.parquet}
  PASSRATE_MIN=${PASSRATE_MIN:-0.6}
  MAX_ITER=${MAX_ITER:-10}
  VAL_PATIENCE=${VAL_PATIENCE:-2}
  FINAL_EPOCHS=${FINAL_EPOCHS:-3}

  export ROOT SAVE_ROOT MODEL_NAME BASE_PARQUET BASE_MODEL ITER_ROOT
  export VAL_PATIENCE VAL_MIN_DELTA MAX_ITER PASSRATE_MIN SLOPE_MIN HIST_LEN
  export RUN_FINAL_LONG_TRAIN FINAL_EPOCHS

  echo "=============================================="
  echo " TraPO v2 full pipeline"
  echo " BASE_MODEL=$BASE_MODEL"
  echo " BASE_PARQUET=$BASE_PARQUET"
  echo " ITER_ROOT=$ITER_ROOT"
  echo " VAL_PATIENCE=$VAL_PATIENCE  PASSRATE_MIN=$PASSRATE_MIN"
  echo " RUN_FINAL_LONG_TRAIN=$RUN_FINAL_LONG_TRAIN"
  echo "=============================================="
  echo ""
  echo ">>> Phase 1: iterative warmup + probe + promote"

  trapo_v2_cmd_loop "$@"

  if [[ ! -f "$FINAL_PARQUET" ]]; then
    echo "ERROR: missing $FINAL_PARQUET after loop"
    exit 1
  fi

  trapo_v2_print_final_counts "$FINAL_PARQUET"

  if [[ "$RUN_FINAL_LONG_TRAIN" == "1" ]]; then
    echo ""
    echo ">>> Phase 2: long train on final labeled-expanded parquet"
    export TRAIN_PARQUET=$FINAL_PARQUET
    export INIT_MODEL=$BASE_MODEL
    set -x
    trapo_v2_run_final "$@"
  else
    echo ""
    echo ">>> Phase 2 skipped (RUN_FINAL_LONG_TRAIN=0)"
  fi

  echo ""
  echo "Pipeline done."
  echo "  loop state:  $ITER_ROOT/loop_state.json"
  echo "  final data:  $FINAL_PARQUET"
  echo "  final ckpt:  $SAVE_ROOT/DE-Venus/ckpt/$MODEL_NAME/$FINAL_EXP_NAME"
  echo "  val logs:    $ITER_ROOT/iter_*/log/valid/global_step_*.log"
}

CMD=${1:-help}
if [[ "$CMD" == "help" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  trapo_v2_help
  exit 0
fi
shift || true

trapo_v2_init "$SCRIPT_DIR"

case "$CMD" in
  train)
    set -x
    trapo_v2_run_train "$@"
    ;;
  probe)
    set -x
    trapo_v2_run_probe "$@"
    ;;
  final)
    set -x
    trapo_v2_run_final "$@"
    ;;
  promote)
    if [[ $# -ge 2 ]]; then
      OUTPUT_PARQUET=${3:-${OUTPUT_PARQUET:-$2/stage2_train.parquet}}
      trapo_v2_promote "$1" "$2" "$OUTPUT_PARQUET" "${PROMOTION_ITERATION:-}"
    elif [[ -n "${ITER:-}" ]]; then
      trapo_v2_promote_iter
    else
      SOURCE_PARQUET=${SOURCE_PARQUET:-$DEPT_ROOT/data/dapo-math-14k/semi-supervised-tiny-fixed.parquet}
      PROBE_DIR=${PROBE_DIR:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/semi_supervised_trapo_v2_probe_artifacts/probe_dumps}
      OUTPUT_PARQUET=${OUTPUT_PARQUET:-$PROBE_DIR/stage2_train.parquet}
      trapo_v2_promote "$SOURCE_PARQUET" "$PROBE_DIR" "$OUTPUT_PARQUET" "${PROMOTION_ITERATION:-}"
    fi
    ;;
  loop)
    trapo_v2_cmd_loop "$@"
    ;;
  pipeline)
    trapo_v2_cmd_pipeline "$@"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    trapo_v2_help >&2
    exit 1
    ;;
esac
