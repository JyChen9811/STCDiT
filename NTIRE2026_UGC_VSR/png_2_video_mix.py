import os
import sys
from glob import glob
from PIL import Image
from tqdm import tqdm

from utils import save_video


images_lq_root = "./Video/LQ_images"
images_enhanced_root = "./Video/enhanced_images"


save_dir = "./Video/mix_enhanced_vidoes"
os.makedirs(save_dir, exist_ok=True)

fps = 24
quality = 9
interval = 25  # 每隔25张取一张增强图


subfolders = sorted([
    d for d in os.listdir(images_lq_root)
    if os.path.isdir(os.path.join(images_lq_root, d))
])

for folder_name in tqdm(subfolders, desc="Converting folders"):
    lq_folder = os.path.join(images_lq_root, folder_name)
    enhanced_folder = os.path.join(images_enhanced_root, folder_name)

    lq_files = sorted(glob(os.path.join(lq_folder, "*.png")))

    if len(lq_files) == 0:
        print(f"Skip empty folder: {folder_name}")
        continue

    if not os.path.isdir(enhanced_folder):
        print(f"Skip folder (no enhanced): {folder_name}")
        continue

    frames = []
    for i, lq_path in enumerate(lq_files):
        filename = os.path.basename(lq_path)
        if i % interval == 0:

            enhanced_path = os.path.join(enhanced_folder, filename)
            if os.path.exists(enhanced_path):
                frames.append(Image.open(enhanced_path))
            else:
                print(f"Warning: enhanced not found {enhanced_path}, using lq")
                frames.append(Image.open(lq_path))
        else:

            frames.append(Image.open(lq_path))


    enhanced_count = sum(1 for i in range(len(lq_files)) if i % interval == 0)
    print(f"{folder_name}: {len(frames)} frames total, {enhanced_count} enhanced frames")


    save_path = os.path.join(save_dir, f"{folder_name}.mp4")
    print(f"Saving {len(frames)} frames -> {save_path}")
    save_video(frames, save_path, fps=fps, quality=quality)
    print(f"Done: {save_path}")

print("All done!")