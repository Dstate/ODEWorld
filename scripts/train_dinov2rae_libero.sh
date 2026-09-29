#!/bin/bash
set -e


# settings
PORT=28896
NUM_PROCS=8
AVAILABLE_GPUS="0,1,2,3,4,5,6,7"
META_FILE_PATH="assets/metas/libero_train.json"
TAST_NAME="dinov2rae_libero"

# base settings
INITIAL_SEED=42
CONDA_ENV_NAME="odeworld"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME" || exit 1
OUTPUT_ROOT="runnings/"

for (( run=0; run < 1; run++ )); do
    SEED=$(( INITIAL_SEED + run ))
    OUTPUT_DIR="$OUTPUT_ROOT/$TAST_NAME/$SEED"
    # training
    CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS torchrun \
        --nproc_per_node=$NUM_PROCS \
        --master_port=$PORT \
        train_dinov2rae.py \
            --meta_file_path $META_FILE_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED
done

echo "===== All processes completed ====="
exit 0
