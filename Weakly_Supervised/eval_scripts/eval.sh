#!/bin/bash

ROOT=..
SAVE_ROOT=..

GROUP=0
TASK=math
EXP_NAME=semi_supervised
TEMPLATE=qwen
BASE_MODEL=Qwen3-8B-Base
MAX_TOKENS=4096

DATA=$ROOT/data/valid.$TASK.parquet

GLOBAL_STEPS=(100 200 300)

if [[ "$GROUP" == "0" ]]; then
    GPUS=0,1,2,3
else
    GPUS=4,5,6,7
fi

for GLOBAL_STEP in "${GLOBAL_STEPS[@]}"; do
    OUTPUT_DIR=$SAVE_ROOT/results/$BASE_MODEL/$EXP_NAME/$GLOBAL_STEP/$TASK/$GROUP
    MODEL_PATH=$SAVE_ROOT/ckpt/$BASE_MODEL/$EXP_NAME/global_step_$GLOBAL_STEP/actor/huggingface

    mkdir -p $OUTPUT_DIR

    echo "Evaluating GLOBAL_STEP=$GLOBAL_STEP ..."

    CUDA_VISIBLE_DEVICES=$GPUS python $ROOT/eval_scripts/generate_vllm.py \
        --model_path $MODEL_PATH \
        --input_file $DATA \
        --remove_system True \
        --add_oat_evaluate True \
        --max_tokens $MAX_TOKENS \
        --output_file $OUTPUT_DIR/$TASK.jsonl \
        --template $TEMPLATE > $OUTPUT_DIR/$TASK.log
done

python /ossfs/workspace/aml0/485004/test.py --gpus $GPUS