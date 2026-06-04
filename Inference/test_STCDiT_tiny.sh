CUDA_VISIBLE_DEVICES=0 python ./Inference/test_STCDiT_tiny.py \
--wan_model_path=./model_checkpoints/Wan2.1-T2V-1.3B \
--stcdit_model_path=./model_checkpoints/STCDiT/tiny_8k.bin \
--video_path=./Video_data/lq \
--caption_path=./Video_data/caption \
--save_path=./Video_data/STCDiT_tiny_enhance_results \
--upscale_factor=1