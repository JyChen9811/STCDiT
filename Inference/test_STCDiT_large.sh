CUDA_VISIBLE_DEVICES=0 python /home/notebook/data/group/cjy/STCDiT/STCDiT/Inference/test_STCDiT_large.py \
--wan_model_path=./model_checkpoints/Wan2.1-I2V-14B-720P \
--stcdit_model_path=./model_checkpoints/STCDiT/large_5k.bin \
--video_path=./Video_data/lq \
--caption_path=./Video_data/caption \
--save_path=./Video_data/STCDiT_large_enhance_results \
--upscale_factor=1