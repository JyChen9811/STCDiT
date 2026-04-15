import torch
import sys
sys.path.append('/home/notebook/data/group/cjy/STCDiT-main')
from diffsynth import ModelManager, save_video, VideoData
from modelscope import snapshot_download
from diffsynth.pipelines.wan_video_i2v_large import WanVideoPipeline_i2v_large
import os
from glob import glob
import json
from peft import LoraConfig, inject_adapter_in_model
from decord import VideoReader, cpu
from PIL import Image
import cv2
import numpy as np
from typing import List
import imageio
from tqdm import tqdm
from video_motion_detection import analyze_video_motion
from utils import load_video, save_video
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="STCDiT inference script")

    parser.add_argument("--wan_model_path", type=str, default="./model_checkpoints/Wan2.1-I2V-14B-720P")
    parser.add_argument("--stcdit_model_path", type=str, default="./model_checkpoints/STCDiT/large_5k.bin")

    parser.add_argument("--video_path", type=str, default="./VideoLQ/lq")
    parser.add_argument("--caption_path", type=str, default="./VideoLQ/caption")
    parser.add_argument("--save_path", type=str, default="./VideoLQ/results")

    parser.add_argument("--minimal_segment_length", type=int, default=9)
    parser.add_argument("--max_process_width", type=int, default=1280)
    parser.add_argument("--max_process_height", type=int, default=960)
    parser.add_argument("--tile_frame_kernel", type=int, default=21)

    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--upscale_factor", type=int, default=2)

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，杂乱的背景",
    )
    return parser.parse_args()


args = parse_args()




model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        [
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00001-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00002-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00003-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00004-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00005-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00006-of-00007.safetensors"),
            os.path.join(args.wan_model_path, "diffusion_pytorch_model-00007-of-00007.safetensors"),        
        ],
        os.path.join(args.wan_model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(args.wan_model_path, "Wan2.1_VAE.pth"),
    ],
    torch_dtype=torch.bfloat16,
)

pipe = WanVideoPipeline_i2v_large.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.denoising_model_enable_lq_input()

lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    init_lora_weights=False,
    target_modules="q,k,v,o,ffn.0,ffn.2".split(","),
)

pipe.denoising_model_load_lora(lora_config)

state_dict = torch.load(args.stcdit_model_path, map_location='cpu')
replace_state_dict = {
    k.replace('pipe.dit.', '', 1): v.detach().clone().cpu() for k, v in state_dict.items()
    if isinstance(v, torch.Tensor)
}
pipe.denoising_model().load_state_dict(replace_state_dict, strict=False)
pipe.denoising_model().to(dtype=torch.bfloat16)
pipe.enable_cpu_offload()



Video_path = args.video_path
Caption_path = args.caption_path
save_path = args.save_path

video_files = sorted(glob(os.path.join(Video_path, "**", "*.mp4"), recursive=True))
caption_files = sorted(glob(os.path.join(Caption_path, "**", "*.json"), recursive=True))

os.makedirs(save_path, exist_ok=True)


exist_file = glob(os.path.join(save_path, "**", "*.mp4"), recursive=True)

for video_file_path, caption_file_path in zip(video_files, caption_files):
    print(f"Processing: {video_file_path}, {caption_file_path}")
    if os.path.join(save_path, os.path.basename(video_file_path)) in exist_file:
        print(f"Exist: {video_file_path}, {caption_file_path}")
        continue

    video, vr, target_size = load_video(video_file_path, upscale_factor=args.upscale_factor)
    segments = analyze_video_motion(
        video_file_path,
        minimal_segment_length=args.minimal_segment_length
    )

    json_file = json.load(open(caption_file_path))
    text = json_file["caption"]

    width, height = video[0].size
    num_frames = len(video)
    
    tile_frame_kernel = args.tile_frame_kernel

    if width < args.max_process_width:
        width_kernel = width // 8
    else:
        width_kernel = args.max_process_width // 8

    if height < args.max_process_height:
        height_kernel = height // 8
    else:
        height_kernel = args.max_process_height // 8
    
    negative_prompt = args.negative_prompt
    
    video = pipe.test_tlc_seg(
        input_lq=video,
        segments=segments,
        prompt=text,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        width=width,
        height=height,
        num_frames=num_frames,
        tiled=True,
        tile_kernel=(tile_frame_kernel, height_kernel, width_kernel),
        seed=42
    )

    save_video(video, os.path.join(save_path, os.path.basename(video_file_path)), fps=24, quality=9, target_size=target_size)