#!/bin/bash

# GPU_TYPE="h100"
GPU_TYPE="smc6kd"
# NSYS_CAPTURE_RANGE="pipeline"
NSYS_CAPTURE_RANGE="monitor_window"

# MODEL_BASE="/scratch/HunyuanVideo/ckpts"
MODEL_BASE=""

# 创建nsys输出目录
NSYS_OUTPUT_DIR="nsys/smc521.6kd.fp8_sage_gemm"
mkdir -p $NSYS_OUTPUT_DIR

# 定义配置数组
# 格式：VIDEO_SIZE,VIDEO_LENGTH,NGPU
configs=(
    "640,61,1"
    "960,61,1"
    "960,129,1"
    "1408,129,1"
    "640,61,8"
    "960,61,8"
    "960,129,8"
    "1408,129,8"
)

# 定义模型映射
declare -A model_mapping
model_mapping["HYVideo-T/2-SingleDual-2to1-NoMod"]="s40d20"
model_mapping["HYVideo-T/2-PureDual-NoMod"]="d20"

# 遍历所有配置和模型组合
for config in "${configs[@]}"; do
    # 解析配置
    IFS=',' read -r VIDEO_SIZE VIDEO_LENGTH NGPU <<< "$config"

    # 遍历所有模型
    for MODEL in "${!model_mapping[@]}"; do
        MODEL_NAME="${model_mapping[$MODEL]}"

        if [ $VIDEO_SIZE == "1408" ]; then
            INFER_STEPS=7
        else
            INFER_STEPS=10
        fi

        if [ $NGPU -eq 1 ]; then
            echo "正在运行配置: VIDEO_SIZE=$VIDEO_SIZE, VIDEO_LENGTH=$VIDEO_LENGTH, NGPU=$NGPU, MODEL=$MODEL, MODEL_NAME=$MODEL_NAME INFER_STEPS=$INFER_STEPS"

            # 构建输出文件名
            output_file="${NSYS_OUTPUT_DIR}/hunyuan.${NSYS_CAPTURE_RANGE}.${MODEL_NAME}.${VIDEO_SIZE}x${VIDEO_SIZE}x${VIDEO_LENGTH}.sp${NGPU}.${GPU_TYPE}"
            # 运行nsys profile命令
            nsys profile -o "$output_file" -f true \
                         --capture-range=nvtx \
                         --nvtx-capture="${NSYS_CAPTURE_RANGE}" \
                         --capture-range-end=stop \
                         --cpuctxsw=none \
                         --sample=none \
                         bash quick_run.sh "$NGPU" "$VIDEO_SIZE" "$VIDEO_LENGTH" "$INFER_STEPS" \
                         "$MODEL" "$MODEL_BASE"
            # 检查命令执行状态
            if [ $? -eq 0 ]; then
                echo "✅ 成功完成: $output_file"
            else
                echo "❌ 执行失败: $output_file"
            fi
        else   
            echo "正在运行配置: VIDEO_SIZE=$VIDEO_SIZE, VIDEO_LENGTH=$VIDEO_LENGTH, NGPU=$NGPU, MODEL=$MODEL, MODEL_NAME=$MODEL_NAME, INFER_STEPS=$INFER_STEPS"
            output_file="${NSYS_OUTPUT_DIR}/hunyuan.${NSYS_CAPTURE_RANGE}.${MODEL_NAME}.${VIDEO_SIZE}x${VIDEO_SIZE}x${VIDEO_LENGTH}.sp${NGPU}.${GPU_TYPE}"
            nsys profile -o "$output_file" -f true \
                         --capture-range=nvtx \
                         --nvtx-capture="${NSYS_CAPTURE_RANGE}" \
                         --capture-range-end=stop \
                         --cpuctxsw=none \
                         --sample=none \
                         bash quick_run.sh "$NGPU" "$VIDEO_SIZE" "$VIDEO_LENGTH" "$INFER_STEPS" \
                         "$MODEL" "$MODEL_BASE"
            # 检查命令执行状态
            if [ $? -eq 0 ]; then
                echo "✅ 成功完成: $output_file"
            else
                echo "❌ 执行失败: $output_file"
            fi
        fi
        echo "----------------------------------------"
    done
done

echo "所有profile任务完成!"
