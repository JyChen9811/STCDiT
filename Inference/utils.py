from decord import VideoReader, cpu
from PIL import Image
import cv2
import numpy as np
from typing import List
import imageio
from tqdm import tqdm

def load_video(video_path, upscale_factor=2):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_num_frames = len(vr)

    images_group = list()
    
    for frame_index in range(total_num_frames):
        img = Image.fromarray(vr[frame_index].asnumpy())

        w, h = img.size
        target_w = w * upscale_factor
        target_h = h * upscale_factor
        target_size = (target_w, target_h)
        img = img.resize((target_w, target_h), Image.LANCZOS)

        w, h = img.size
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16

        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        images_group.append(img)
    
    return images_group, vr, target_size
    


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None, target_size=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, macro_block_size=None, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        if target_size is not None:
            frame = np.array(Image.fromarray(frame).resize(target_size, Image.LANCZOS))
        writer.append_data(frame)
    writer.close()


def pad_input_lq(input_lq):
    num_frames = len(input_lq)
    pad_frames = 0

    if num_frames % 4 != 1:
        target_frames = (num_frames + 2) // 4 * 4 + 1
        pad_frames = target_frames - num_frames
        last_frame = input_lq[-1]
        input_lq.extend([last_frame] * pad_frames)

    return input_lq, pad_frames