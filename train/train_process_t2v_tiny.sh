
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python ./train/train_wan_t2v_tiny.py \
  --gpu_num 8 \
  --task train \
  --train_architecture lora \
  --dataset_path data/example_dataset \
  --output_path ./output/t2v_tiny \
  --training_strategy deepspeed_stage_2 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --dit_path "./model_checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 8000 \
  --max_epochs 1 \
  --learning_rate 5e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --val_check_interval 500