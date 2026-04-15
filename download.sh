### checkpoints
modelscope download junyangchen/STCDiT_ckpt --local_dir=./model_checkpoints/STCDiT
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir=./model_checkpoints/Wan2.1-I2V-14B-720P
modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir=./model_checkpoints/Wan2.1-T2V-1.3B
## export HF_ENDPOINT=https://hf-mirror.com
huggingface download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./model_checkpoints/Qwen-VL --local-dir-use-symlinks false


### validation dataset and enhancement results
modelscope download junyangchen/SportsLQ --local_dir=./dataset/SportsLQ --repo-type=dataset
modelscope download junyangchen/STCDiT_large_results --local_dir=./results/enhancement_results --repo-type=dataset


