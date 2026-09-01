# Step 2: Attention probing + uncertainty scoring
# 1) Identify global attention heads (single GPU)
# 2) Score each question in parallel across N_GPUS processes
# 3) Merge per-split results into scored.jsonl

ROOT=..

N_GPUS=8
N_PER_PROMPT=8
DATASET=dapo-math-14k
BASE_MODEL=Qwen3-4B-Base

MODEL_PATH=..
INPUT_PATH=$ROOT/DE-Venus/data/gen/$DATASET/$BASE_MODEL/responses.jsonl
OUTPUT_PATH=$ROOT/DE-Venus/data/gen/$DATASET/$BASE_MODEL/scored.jsonl
GLOBAL_ATTN_PATH=$ROOT/DE-Venus/data/gen/$DATASET/$BASE_MODEL/global_attn_heads.pkl

# Attention probing (single GPU)
echo "Step 1: Attention probing..."
python3 $ROOT/DE-Venus/Weakly_Supervised/Research/Data_Selection/attn_probing.py \
  --model_path $MODEL_PATH \
  --input_path $INPUT_PATH \
  --output_path $GLOBAL_ATTN_PATH \
  --n_per_prompt $N_PER_PROMPT \
  --H_low 20 \
  --H_high 100

# Parallel scoring (one process per GPU)
SPLIT_DIR=$(dirname $OUTPUT_PATH)/splits
mkdir -p $SPLIT_DIR

echo "Step 2: Launching $N_GPUS scoring processes..."
for i in $(seq 0 $((N_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 $ROOT/DE-Venus/Weakly_Supervised/Research/Data_Selection/score_data.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_PATH \
    --output_path $SPLIT_DIR/split${i}.jsonl \
    --global_attn_path $GLOBAL_ATTN_PATH \
    --split $i \
    --n_gpus $N_GPUS \
    --n_per_prompt $N_PER_PROMPT \
    --H_low 20 \
    --H_high 100 &
done

echo "Waiting for all processes to finish..."
wait
echo "All processes finished."

# Merge splits
echo "Merging results..."
> $OUTPUT_PATH
for i in $(seq 0 $((N_GPUS - 1))); do
  cat $SPLIT_DIR/split${i}.jsonl >> $OUTPUT_PATH
done

echo "Merged result saved to $OUTPUT_PATH"
rm -rf $SPLIT_DIR
echo "Done."
