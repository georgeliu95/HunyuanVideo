#!/bin/bash

NNODES=$1
NPROC_PER_NODE=$2
ULYSSES_DEGREE=$3
JOB_ID=$4
MASTER_ADDR=$5
MASTER_PORT=$6
USE_NSYS=$7


if [ "$ULYSSES_DEGREE" == 1 ]; then
    NNODES=1
    NPROC_PER_NODE=1
fi

if [ "$USE_NSYS" == "true" ]; then
    NSYS_CMD="nsys profile \
                -f true \
                --capture-range=nvtx \
                --nvtx-capture=pipeline \
                --capture-range-end=stop \
                --cpuctxsw=none \
                --sample=none"
else
    NSYS_CMD=""
fi

resolution_configs=(
    "640,61"
    "960,61"
    "960,129"
    "1408,129"
)
declare -A model_mapping
model_mapping["HYVideo-T/2-SingleDual-2to1-NoMod"]="s40d20"
model_mapping["HYVideo-T/2-PureDual-NoMod"]="d20"

TORCHRUN_CMD="torchrun \
                --nnodes=$NNODES \
                --nproc_per_node=$NPROC_PER_NODE \
                --rdzv_id=$JOB_ID \
                --rdzv_backend=c10d \
                --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

for resolution_config in "${resolution_configs[@]}"; do
    IFS=',' read -r VIDEO_SIZE VIDEO_LENGTH <<< "$resolution_config"
    for model in "${!model_mapping[@]}"; do
        MODEL_NAME="${model_mapping[$model]}"
        if [ $VIDEO_SIZE == "1408" ]; then
            INFER_STEPS=7
        else
            INFER_STEPS=10
        fi
        SAMPLE_CMD="sample_video_custom.py \
                        --video-size ${VIDEO_SIZE} ${VIDEO_SIZE} \
                        --video-length ${VIDEO_LENGTH} \
                        --infer-steps ${INFER_STEPS} \
                        --cfg-scale 0.7 \
                        --flow-reverse \
                        --seed 42 \
                        --ulysses-degree ${ULYSSES_DEGREE} \
                        --ring-degree 1 \
                        --save-path ./results \
                        --vae-tiling \
                        --skip-load-model \
                        --warmup \
                        --attn-type fa \
                        --optimize-memcpy \
                        --model ${model}"
        if [ "$ULYSSES_DEGREE" -gt 1 ]; then
            SAMPLE_CMD="${TORCHRUN_CMD} ${SAMPLE_CMD}"
        fi
        if [ "$USE_NSYS" == "true" ]; then
            NSYS_REPORT_PATH="nsys/smc521.6kd.all/hyvideo.pipeline.${MODEL_NAME}.${VIDEO_SIZE}x${VIDEO_LENGTH}.sp${ULYSSES_DEGREE}.${SLURMD_NODENAME}.6kd"
            NSYS_CMD="nsys profile \
                        -f true \
                        -o ${NSYS_REPORT_PATH} \
                        --capture-range=nvtx \
                        --nvtx-capture=pipeline \
                        --capture-range-end=stop \
                        --cpuctxsw=none \
                        --sample=none"
        fi
        SCENARIO_CMD="${NSYS_CMD} ${SAMPLE_CMD}"
        echo "[INFO] Executing: $SCENARIO_CMD"
        $SCENARIO_CMD
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed, file saved as: $NSYS_REPORT_PATH"
        else
            echo "❌ Failed to execute, Configuration: ${MODEL_NAME}.${VIDEO_SIZE}x${VIDEO_LENGTH}.sp${ULYSSES_DEGREE}.${SLURMD_NODENAME}"
        fi
    done
done

