DIT_DIR="./model_checkpoints/Wan2.1-I2V-14B-720P"
DIT_PATH=$(printf "${DIT_DIR}/diffusion_pytorch_model-%05d-of-00007.safetensors," $(seq 1 7))
DIT_PATH=${DIT_PATH%,}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python ./train/train_wan_i2v_large.py \
  --gpu_num 8 \
  --task train \
  --train_architecture lora \
  --dataset_path data/example_dataset \
  --output_path ./output/i2v_large \
  --training_strategy deepspeed_stage_2 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --dit_path "$DIT_PATH" \
  --steps_per_epoch 5000 \
  --max_epochs 1 \
  --learning_rate 5e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --val_check_interval 1000 
  