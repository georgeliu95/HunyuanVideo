#!/bin/bash

NGPU=$1
VIDEO_SIZE=$2
VIDEO_LENGTH=$3
INFER_STEPS=$4
MODEL=$5
MODEL_BASE=$6

# 设置默认值
if [ -z "$INFER_STEPS" ]; then
    INFER_STEPS=5
fi

if [ -z "$MODEL" ]; then
    MODEL="HYVideo-T/2-SingleDual-2to1-NoMod"
fi

if [ -z "$MODEL_BASE" ]; then
    MODEL_BASE="ckpts"
fi

export MODEL_BASE=$MODEL_BASE

ULYSSES_DEGREE=$NGPU
RING_DEGREE=1

if [ $NGPU -eq 1 ]; then
    echo "未使用分布策略"
    echo "ulysses-degree: $ULYSSES_DEGREE, ring-degree: $RING_DEGREE"
else
    echo "使用分布策略: ulysses"
    echo "ulysses-degree: $ULYSSES_DEGREE, ring-degree: $RING_DEGREE"
fi

CMD="torchrun --nproc_per_node=${NGPU} sample_video_custom.py \
    --video-size ${VIDEO_SIZE} ${VIDEO_SIZE} --video-length ${VIDEO_LENGTH} \
    --infer-steps ${INFER_STEPS} \
    --prompt \"A cat walks on the grass, realistic style.\" \
    --neg-prompt \"Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards.\" \
    --cfg-scale 0.7 \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree ${ULYSSES_DEGREE} \
    --ring-degree ${RING_DEGREE} \
    --save-path ./results \
    --vae-tiling \
    --skip-load-model \
    --warmup \
    --optimize-memcpy \
    --model ${MODEL} \
    --model-base ${MODEL_BASE} \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
echo "正在执行命令: "
echo $CMD
eval $CMD
