CUDA_VISIBLE_DEVICES=0 python /home/notebook/code/personal/S9062668/STCDiT/Inference/test_STCDiT.py \
--wan_model_path=./model_checkpoints/Wan2.1-T2V-1.3B \
--stcdit_model_path=./model_checkpoints/STCDiT/tiny_8k.bin \
--video_path=./Video_data/lq \
--caption_path=./Video_data/caption \
--save_path=./Video_data/STCDiT_tiny_enhance_results \
--upscale_factor=1