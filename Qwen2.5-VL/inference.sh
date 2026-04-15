
# For Video
torchrun --nproc_per_node=8 --master_port=8399 ./STCDiT/Qwen2.5-VL-main/qwen_batch_video_ddp_input_path.py --video_file_path=./Video_data/video --json_folder=./Video_data/caption

