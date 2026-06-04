import torch
import sys
sys.path.append('/home/notebook/data/group/cjy/STCDiT-main')
from diffsynth import ModelManager, save_video, VideoData
from modelscope import snapshot_download
from diffsynth.pipelines.wan_video_t2v_tiny import WanVideoPipeline_t2v_tiny
import os
from glob import glob
from peft import LoraConfig, inject_adapter_in_model
import ipdb
# Load models
model_manager = ModelManager(device="cpu")
# model_manager.load_models(
#     ["/gemini/code/cjy/Checkpoints/14B_I2V_720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
#     torch_dtype=torch.float32, # Image Encoder is loaded with float32
# )
model_manager.load_models(
    [
        "./model_checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)

pipe = WanVideoPipeline_t2v_tiny.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")


pipe.enable_cpu_offload()


def expand_pattern(lst):
    result = []
    for n in lst:
        result.append(1)
        result.extend([0] * (n - 1))
    return result

def balance_ones_with_indices(lists):
    preprocessed = []
    for lst in lists:
        fixed = lst[:]
        if fixed.count(1) == 0:
            fixed[0] = 1  # 插入一个1
        preprocessed.append(fixed)

    one_counts = [lst.count(1) for lst in preprocessed]
    min_ones = min(one_counts)

    balanced = []
    retained_indices = []  # 记录每个列表中保留的1的位置

    for lst in preprocessed:
        current = lst[:]
        ones_idx = [i for i, v in enumerate(current) if v == 1]

        if len(ones_idx) > min_ones:
            to_zero = random.sample(ones_idx, len(ones_idx) - min_ones)
            for idx in to_zero:
                current[idx] = 0
            retained = [i for i in ones_idx if i not in to_zero]
        else:
            retained = ones_idx

        balanced.append(current)
        retained_indices.append(retained)

    return balanced, retained_indices
path = './STCDiT-main/output/t2v_tiny'
save_dir = './STCDiT-main/output/t2v_tiny'
pth_files = glob(os.path.join(path, "**", "*.pth"), recursive=True)

exist_file = glob(os.path.join(save_dir, "**", "*.mp4"), recursive=True)
for file_path in pth_files:
    if file_path.replace('.pth', '.mp4') in exist_file:
        continue


    tmp = torch.load(file_path, map_location="cpu")
    lq_latents = tmp['fake_latents'].to("cuda")
    key_frame_idx = tmp['video_chunk_idx']
    key_idx = [expand_pattern(key_frame_idx)]
    key_idx_list, retained_indices = balance_ones_with_indices(key_idx)
    video = pipe.test_decode_seg(latents=lq_latents, segments=key_frame_idx)
    save_video(video, os.path.join(save_dir, os.path.basename(file_path).replace('.pth', '.mp4')), fps=15, quality=9)