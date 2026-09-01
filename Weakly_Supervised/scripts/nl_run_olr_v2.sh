#!/usr/bin/env bash
# nl_run_olr_v2: pipeline | loop | train | relabel | final
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=olr_v2_lib.sh
source "$SCRIPT_DIR/olr_v2_lib.sh"

nl_olr_v2_cmd_loop() {
  TRAIN_PARQUET=$BASE_PARQUET
  BEST_VAL=""
  VAL_STALL=0
  STATE_FILE=$ITER_ROOT/loop_state.json
  FINAL_PARQUET=$ITER_ROOT/final_train.parquet
  mkdir -p "$ITER_ROOT"

  if [[ "${ENSURE_ALL_LABELED:-1}" == "1" ]]; then
    python3 -c "
from Weakly_Supervised.utils.analysis.build_noisy_label_dataset import ensure_noisy_label_parquet_all_labeled
import json
stats = ensure_noisy_label_parquet_all_labeled('${BASE_PARQUET}')
print('ensure_all_labeled:', json.dumps(stats))
"
  fi

  echo "=== NL-OLR v2 iterative loop (warmup + probe + relabel) ==="
  echo "BASE_MODEL=$BASE_MODEL"
  echo "BASE_PARQUET=$BASE_PARQUET"
  echo "ITER_ROOT=$ITER_ROOT"
  echo "PASSRATE_MIN=$PASSRATE_MIN  SLOPE_MIN=$SLOPE_MIN  SLOPE_MARGIN=$SLOPE_MARGIN"

  export NL_OLR_WARMUP_STEPS_BASE=${NL_OLR_WARMUP_STEPS_BASE:-$NL_OLR_WARMUP_STEPS}
  export NL_OLR_PROBE_INTERVAL_BASE=${NL_OLR_PROBE_INTERVAL_BASE:-$NL_OLR_PROBE_INTERVAL}
  N_ROWS_PREV=

  for ITER in $(seq 0 "$MAX_ITER"); do
    echo "---------- iteration $ITER ----------"
    COUNTS=$(python3 -c "
from Weakly_Supervised.utils.analysis.build_stage2_dataset import count_labeled_unlabeled_rows
import json
print(json.dumps(count_labeled_unlabeled_rows('${TRAIN_PARQUET}')))
")
    N_TOTAL=$(echo "$COUNTS" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_total'])")
    echo "train_parquet=$TRAIN_PARQUET  counts=$COUNTS"
    nl_olr_v2_apply_scaled_probe_schedule "$N_TOTAL"

    export ITER TRAIN_PARQUET BASE_MODEL ITER_ROOT ROOT SAVE_ROOT MODEL_NAME EXP_NAME
    unset CKPT_DIR LOG_DIR NL_OLR_CKPT_DIR NL_OLR_LOG_DIR
    LOG_DIR=$ITER_ROOT/iter_${ITER}/log
    PROBE_DIR=$LOG_DIR/probe_dumps
    set +e
    set -x
    nl_olr_v2_run_train "$@"
    TRAIN_EXIT=$?
    set -e
    set +x
    if [[ "$TRAIN_EXIT" -ne 0 ]]; then
      if [[ -f "$PROBE_DIR/all_probes_summary.jsonl" ]]; then
        echo "WARN: training exited $TRAIN_EXIT but probe dump exists; continuing relabel."
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
    nl_olr_v2_relabel "$TRAIN_PARQUET" "$PROBE_DIR" "$NEXT_PARQUET" "$ITER"

    META=$ITER_ROOT/iter_${ITER}/train_next.advance_meta.json
    N_RELABEL=$(python3 -c "import json; print(json.load(open('$META'))['n_relabeled'])")
    N_SAME=$(python3 -c "import json; print(json.load(open('$META'))['n_candidate_same_gt'])")

    echo "iteration=$ITER  val_average=$VAL_AVG  val_step=$VAL_STEP  relabeled=$N_RELABEL  same_gt_skipped=$N_SAME"

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
    "n_rows": int("$N_TOTAL"),
    "n_relabeled": int("$N_RELABEL"),
    "n_candidate_same_gt": int("$N_SAME"),
    "val_average": val_info.get("average"),
    "val_global_step": val_info.get("global_step"),
    "val_per_source": val_info.get("per_source"),
    "relabel_summary": meta.get("relabel_summary"),
}
path = pathlib.Path("$STATE_FILE")
history = json.loads(path.read_text()).get("history", []) if path.exists() else []
history.append(state)
path.write_text(json.dumps({"history": history, "last": state}, indent=2))
PY

    cp -f "$NEXT_PARQUET" "$FINAL_PARQUET"

    if [[ "$N_RELABEL" -eq 0 ]]; then
      echo "No label rewrites this round; stopping."
      break
    fi

    if [[ -z "$VAL_AVG" ]]; then
      echo "WARN: no val average; skip val early-stop this round."
    elif [[ -n "$BEST_VAL" ]]; then
      IMPROVED=$(python3 -c "print(1 if float('$VAL_AVG') > float('$BEST_VAL') + float('$VAL_MIN_DELTA') else 0)")
      if [[ "$IMPROVED" -eq 1 ]]; then
        BEST_VAL=$VAL_AVG
        VAL_STALL=0
      else
        VAL_STALL=$((VAL_STALL + 1))
        if [[ "$VAL_STALL" -ge "$VAL_PATIENCE" ]]; then
          echo "Early stop: val average plateau."
          break
        fi
      fi
    else
      BEST_VAL=$VAL_AVG
    fi

    TRAIN_PARQUET=$NEXT_PARQUET
  done

  echo "Loop finished. final_train_parquet=$FINAL_PARQUET"
  echo "State: $STATE_FILE"
}

nl_olr_v2_cmd_pipeline() {
  FINAL_PARQUET=${FINAL_PARQUET:-$ITER_ROOT/final_train.parquet}
  PASSRATE_MIN=${PASSRATE_MIN:-0.5}
  MAX_ITER=${MAX_ITER:-10}
  VAL_PATIENCE=${VAL_PATIENCE:-2}
  FINAL_EPOCHS=${FINAL_EPOCHS:-3}

  export ROOT SAVE_ROOT MODEL_NAME BASE_PARQUET BASE_MODEL ITER_ROOT
  export VAL_PATIENCE VAL_MIN_DELTA MAX_ITER PASSRATE_MIN SLOPE_MIN SLOPE_MARGIN HIST_LEN
  export RUN_FINAL_LONG_TRAIN FINAL_EPOCHS

  echo ">>> Phase 1: NL-OLR v2 iterative warmup + probe + relabel"
  nl_olr_v2_cmd_loop "$@"

  if [[ ! -f "$FINAL_PARQUET" ]]; then
    echo "ERROR: missing $FINAL_PARQUET"
    exit 1
  fi
  nl_olr_v2_print_final_counts "$FINAL_PARQUET"

  if [[ "$RUN_FINAL_LONG_TRAIN" == "1" ]]; then
    echo ">>> Phase 2: long train on refined parquet"
    export TRAIN_PARQUET=$FINAL_PARQUET INIT_MODEL=$BASE_MODEL
    set -x
    nl_olr_v2_run_final "$@"
  else
    echo ">>> Phase 2 skipped (RUN_FINAL_LONG_TRAIN=0)"
  fi
}

CMD=${1:-help}
if [[ "$CMD" == "help" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  nl_olr_v2_help
  exit 0
fi
shift || true
nl_olr_v2_init "$SCRIPT_DIR"

case "$CMD" in
  train) set -x; nl_olr_v2_run_train "$@" ;;
  final) set -x; nl_olr_v2_run_final "$@" ;;
  relabel)
    if [[ $# -ge 2 ]]; then
      OUTPUT_PARQUET=${3:-${OUTPUT_PARQUET:-$2/train_next.parquet}}
      nl_olr_v2_relabel "$1" "$2" "$OUTPUT_PARQUET" "${RELABEL_ITERATION:-}"
    elif [[ -n "${ITER:-}" ]]; then
      nl_olr_v2_relabel_iter
    else
      nl_olr_v2_relabel "${SOURCE_PARQUET:-$BASE_PARQUET}" "${PROBE_DIR:-}" "${OUTPUT_PARQUET:-}"
    fi
    ;;
  loop) nl_olr_v2_cmd_loop "$@" ;;
  pipeline) nl_olr_v2_cmd_pipeline "$@" ;;
  *) echo "Unknown: $CMD" >&2; nl_olr_v2_help >&2; exit 1 ;;
esac
