WORLD_SIZE=8
SCRIPT=./Inference/test_STCDiT_tiny.py
WAN_MODEL_PATH=/home/notebook/data/personal/S9062668/model_checkpoints/Wan2.1-T2V-1.3B
STCDIT_MODEL_PATH=./model_checkpoints/STCDiT/tiny_8k.bin
VIDEO_PATH=./Video_data/lq 
CAPTION_PATH=./Video_data/caption
SAVE_PATH=./Video_data/STCDiT_tiny_enhance_results
LOG_DIR=${SAVE_PATH}/_logs
mkdir -p $LOG_DIR

for GPU in $(seq 0 $((WORLD_SIZE-1))); do
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --wan_model_path=$WAN_MODEL_PATH \
        --stcdit_model_path=$STCDIT_MODEL_PATH \
        --video_path=$VIDEO_PATH \
        --caption_path=$CAPTION_PATH \
        --cfg_scale=1 \
        --save_path=$SAVE_PATH \
        --upscale_factor=1 \
        > ${LOG_DIR}/gpu${GPU}.log 2>&1 &
    echo "Launched GPU $GPU (pid=$!)"
    sleep 15   # 错峰加载模型，缓解磁盘/CPU IO 抖动；不需要可去掉
done

wait
echo "All GPUs finished."



# kill
# pkill -9 -f 'test_STCDiT_tiny.py'
# pkill -9 -f 'test_STCDiT_tiny.sh'