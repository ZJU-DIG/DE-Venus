# Step 3: Active data selection
# Selects training samples from scored.jsonl based on uncertainty metrics.

ROOT=..

DATASET=dapo-math-14k
BASE_MODEL=Qwen3-4B-Base

INPUT_PATH=$ROOT/DE-Venus/data/gen/$DATASET/$BASE_MODEL/scored.jsonl
OUTPUT_PATH=$ROOT/DE-Venus/data/$DATASET/$BASE_MODEL/selected_train.parquet

# Simulation resources (unlabeled mode only)
SIMULATE_PARQUET=$ROOT/DE-Venus/data/$DATASET/train.parquet
RESPONSES_PATH=$ROOT/DE-Venus/data/gen/$DATASET/$BASE_MODEL/responses.jsonl

# Selection mode: labeled | unlabeled
MODE=unlabeled

# Sorting metric (accuracy is labeled-mode only):
#   pivot_num, consistency, entropy, self_certainty, coe_r, coe_c, cot_kinetics, accuracy
SORT_BY=pivot_num

if [ "$MODE" = "labeled" ]; then
  python3 $ROOT/DE-Venus/Weakly_Supervised/Research/Data_Selection/active_select.py \
    --mode labeled \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --sort_by $SORT_BY \
    --acc_l 0.2 \
    --acc_h 0.7 \
    --train_budget 4000
else
  # Optional: --label_budget N  fixes labeled count (incl. probes); otherwise determined by --acc_a
  #           --train_budget N  fixes total count (labeled + unlabeled); otherwise keeps all
  python3 $ROOT/DE-Venus/Weakly_Supervised/Research/Data_Selection/active_select.py \
    --mode unlabeled \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --sort_by $SORT_BY \
    --acc_l 0.2 \
    --acc_h 0.7 \
    --acc_a 0.3 \
    --probe_k 200 \
    --window_size 20 \
    --simulate \
    --simulate_parquet_path $SIMULATE_PARQUET \
    --responses_path $RESPONSES_PATH
fi
