import cv2
import math
import time
import os
import os.path as osp
import numpy as np
import random
import torch

import yaml
from copy import deepcopy
from pathlib import Path
from torch.utils import data as data
from decord import VideoReader, cpu
from .mmcv_transforms import Clip, UnsharpMasking, RescaleTOOneNegOne
from .mmcv_transforms import RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression, DegradationsWithShuffle
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from .transforms import augment, single_random_crop, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
from glob import glob
from torch.utils.data import DataLoader
import json
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d

CROP_FRAME_WIDTH = 1920
CROP_FRAME_HEIGHT = 1080


def load_video(video_path, num_frames, return_msg=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_num_frames = len(vr)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(transforms(img))
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return images_group, msg
    else:
        return images_group


def crop_frame(frame):
    # 如果是 numpy 类型，转换为 PIL Image
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    w, h = frame.size  # PIL 的尺寸顺序是 (width, height)

    # 等比裁剪区域，使得结果可以 resize 到 1920x1080
    scale = min(w / CROP_FRAME_WIDTH, h / CROP_FRAME_HEIGHT)
    target_w = int(scale * CROP_FRAME_WIDTH)
    target_h = int(scale * CROP_FRAME_HEIGHT)
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2

    # 裁剪并 resize
    cropped = frame.crop((start_x, start_y, start_x + target_w, start_y + target_h))
    resized = cropped.resize((CROP_FRAME_WIDTH, CROP_FRAME_HEIGHT), Image.LANCZOS)

    return resized  # 返回 PIL.Image。如果需要 np.array，可以加 np.array(resized)


def should_crop_frame(frame):
    if isinstance(frame, np.ndarray):
        h, w = frame.shape[:2]
    else:
        w, h = frame.size
    return w > CROP_FRAME_WIDTH and h > CROP_FRAME_HEIGHT



class RandomTranslate:
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, max_period_length, max_translation, scale_range):
        self.max_period_length = max_period_length
        self.max_translation = max_translation
        
        self.scale_range = scale_range
        self.translate_prob = 0.25
        self.zoom_prob = 0.25
    def get_translation_params(self, period_length, mode):
        # mode = 'random' if np.random.rand() >= 0.5 else 'linear'
        if mode == 'linear':
            if np.random.rand() >= 0.5:
                tx_end = np.random.uniform(-self.max_translation, -5)
            else:
                tx_end = np.random.uniform(5, self.max_translation)

            if np.random.rand() >= 0.5:
                ty_end = np.random.uniform(-self.max_translation, -5)
            else:
                ty_end = np.random.uniform(5, self.max_translation)
            
            if tx_end < 0:
                tx_seq = np.linspace(-5, tx_end, period_length)
            else:
                tx_seq = np.linspace(5, tx_end, period_length)

            if ty_end < 0:
                ty_seq = np.linspace(-5, ty_end, period_length)
            else:
                ty_seq = np.linspace(5, ty_end, period_length)

        elif mode == 'random':
            tx_seq = np.random.uniform(-self.max_translation, self.max_translation, size=period_length)
            ty_seq = np.random.uniform(-self.max_translation, self.max_translation, size=period_length)
            # 平滑使得变化不那么突兀（如 σ=1）
            tx_seq = gaussian_filter1d(tx_seq, sigma=1)
            ty_seq = gaussian_filter1d(ty_seq, sigma=1)
        return tx_seq, ty_seq


    def generate_scale_params(self, period_length):
        if np.random.rand() >= 0.5:
            scale_end = np.random.uniform(self.scale_range[0], 0.9)
        else:
            scale_end = np.random.uniform(1.1, self.scale_range[1])

        if scale_end < 1:
            scale_seq = np.linspace(0.9, scale_end, period_length)
        else:
            scale_seq = np.linspace(1.1, scale_end, period_length)
        return scale_seq


    def apply_shake(self, img, tx, ty, angle, scale):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        shaken = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return shaken

    def __call__(self, input_video, num_frames, num_periods):

        tx_list = np.zeros(num_frames)
        ty_list = np.zeros(num_frames)
        angle_list = np.zeros(num_frames)
        scale_list = np.ones(num_frames)


        used_intervals = []

        tries = 0
        max_tries = 1000
        selected = []

        while len(selected) < num_periods and tries < max_tries:
            tries += 1
            period_len = np.random.randint(1, self.max_period_length + 1)
            start = np.random.randint(0, num_frames - period_len + 1)

            # 检查是否重叠
            overlap = False
            for s, e in used_intervals:
                if not (start + period_len <= s or start >= e):  # 有交集
                    overlap = True
                    break

            if not overlap:
                selected.append((start, period_len))
                used_intervals.append((start, start + period_len))

        if len(selected) < num_periods:
            raise RuntimeError("找不到足够的不重叠 shake 周期")

        selected.sort()  # optional
        
        output_selected = []
        for start, period_length in selected:
            mode = None
            if np.random.uniform() > self.translate_prob:
                mode = random.choice(['random', 'linear'])
                tx_seq, ty_seq = self.get_translation_params(period_length, mode)
                tx_list[start:start + period_length] = tx_seq
                ty_list[start:start + period_length] = ty_seq


            if np.random.uniform() > self.zoom_prob:
                scale_seq = self.generate_scale_params(period_length)
                scale_list[start:start + period_length] = scale_seq

            if mode is None or mode == 'linear':
                output_selected.append((start, period_length))

            elif mode == 'random':
                for i in range(period_length):
                    output_selected.append((start + i, 1))


        for frame_idx in range(num_frames):
            shaken = self.apply_shake(input_video[frame_idx], tx_list[frame_idx], ty_list[frame_idx],
                                angle_list[frame_idx], scale_list[frame_idx])
            
            input_video[frame_idx] = shaken


        return input_video, output_selected



def generate_translation_params(period_length, max_translation, mode='linear'):
    if mode == 'linear':
        tx_end = np.random.uniform(-max_translation, max_translation)
        ty_end = np.random.uniform(-max_translation, max_translation)
        tx_seq = np.linspace(0, tx_end, period_length)
        ty_seq = np.linspace(0, ty_end, period_length)
    elif mode == 'random':
        tx_seq = np.random.uniform(-max_translation, max_translation, size=period_length)
        ty_seq = np.random.uniform(-max_translation, max_translation, size=period_length)
        # 平滑使得变化不那么突兀（如 σ=1）
        tx_seq = gaussian_filter1d(tx_seq, sigma=1)
        ty_seq = gaussian_filter1d(ty_seq, sigma=1)
    else:
        raise ValueError(f"未知平移模式: {mode}")

    return tx_seq, ty_seq

def generate_scale_params(period_length, scale_range):
    scale_end = np.random.uniform(scale_range[0], scale_range[1])
    scale_seq = np.linspace(1.0, scale_end, period_length)
    return scale_seq

# @DATASET_REGISTRY.register()
class Zoom_translate_RealVSRRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
    """

    def __init__(self, opt, video_path, csv_path, save_path, num_frame=5, img_size=(480, 832), interval_list=[1]):
        super(Zoom_translate_RealVSRRecurrentDataset, self).__init__()
        self.opt = opt
        self.save_path = save_path
        self.num_frame = num_frame
        self.gt_size = img_size


        with open(video_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        self.path = lines
        assert len(self.path) > 0, f"No MP4 files found in {video_path}"
        # temporal augmentation configs

        self.interval_list = interval_list


        # self.csv_data = pd.read_csv(csv_path)

        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; ')

        # Zoom and Translation
        self.random_RandomTranslate = RandomTranslate(max_period_length=9, max_translation=20, scale_range=(0.8, 1.25))
        # the first degradation
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )

        # the second degradation
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.random_mpeg_2 = RandomVideoCompression(
            params=opt['degradation_2']['random_mpeg']['params'],
            keys=opt['degradation_2']['random_mpeg']['keys']
        )

        # final
        opt['degradation_2']['resize_final']['params']['target_size'] = img_size
        self.resize_final = RandomResize(
            params=opt['degradation_2']['resize_final']['params'],
            keys=opt['degradation_2']['resize_final']['keys']
        )
        self.blur_final = RandomBlur(
            params=opt['degradation_2']['blur_final']['params'],
            keys=opt['degradation_2']['blur_final']['keys']
        )

        # # transforms
        self.usm = UnsharpMasking(
            kernel_size=opt['transforms']['usm']['kernel_size'],
            sigma=opt['transforms']['usm']['sigma'],
            weight=opt['transforms']['usm']['weight'],
            threshold=opt['transforms']['usm']['threshold'],
            keys=opt['transforms']['usm']['keys']
        )
        self.clip = Clip(keys=opt['transforms']['clip']['keys'])
        self.rescale = RescaleTOOneNegOne(keys=opt['transforms']['rescale']['keys'])


    # 通过 clip_id 查询对应的 Detailed Description
    def get_description(self, clip_id):
        result = self.csv_data[self.csv_data['clip_id'] == clip_id]
        if not result.empty:
            return result.iloc[0]['Detailed Description']
        else:
            return f"Clip ID {clip_id} not found."


    def __getitem__(self, index):

        gt_size = self.gt_size
        video_path = self.path[index]


        json_path = video_path.replace("/video/", "/caption/").replace(".mp4", ".json")
        with open(json_path, "r") as f:
            init_text = json.load(f)["caption"]



        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        total_num_frames = len(vr)

        # determine the neighboring frames

        max_interval = total_num_frames // self.num_frame

        if max_interval >= 4:
            interval = random.choice(self.interval_list)
        elif 1<= max_interval < 4:
            interval = random.randint(1, max_interval)
        else:
            out_dict = {'lqs_1': [], 'gts_1': [], 'lqs_2': [], 'gts_2': [], 'lqs_3': [], 'gts_3': [], 'lqs_4': [], 'gts_4': [],'video_shape': [[]], 'text': [[]], 'path_1': [[]], 'path_2': [[]], 'path_3': [[]], 'path_4': [[]], 'segments_1': [], 'segments_2': [], 'segments_3': [], 'segments_4': []}
            return out_dict
        # ensure not exceeding the borders
        # if total_num_frames > self.num_frame * interval:
        start_frame_idx = random.randint(0, total_num_frames - self.num_frame * interval)


        end_frame_idx = start_frame_idx + self.num_frame * interval
        img_gts_list = []
        for frame_idx in range(start_frame_idx, end_frame_idx, interval):
            img_gts_list.append(np.array(vr[frame_idx].asnumpy()))

        if should_crop_frame(img_gts_list[0]):
            img_gts_list_processed = [np.array(crop_frame(frame)) for frame in img_gts_list]
        else:
            img_gts_list_processed = img_gts_list

        out_dict = {}
        # randomly crop
        # img_gts = single_random_crop(img_gts, gt_size)
        img_gts = deepcopy(img_gts_list_processed)
        num_periods = random.randint(1, 3)
        img_gts, selected_1 = self.random_RandomTranslate(img_gts, len(img_gts), num_periods=num_periods)
        img_gts = single_random_crop(img_gts, gt_size)

        

        
        cursor = 0
        segments_1 = []
        for idx, length in selected_1:
            if cursor < idx:
                segments_1.append((cursor, idx))          # before the selected region
            segments_1.append((idx, idx + length))        # selected region
            cursor = idx + length

        if cursor < self.num_frame:
            segments_1.append((cursor, self.num_frame))     # remaining tail

        img_lqs = deepcopy(img_gts)

        process_dict = {'lqs': img_lqs, 'gts': img_gts}

        # out_dict = self.usm.transform(out_dict)

        # ## the first degradation
        process_dict = self.random_blur_1(process_dict)
        process_dict = self.random_resize_1(process_dict)
        process_dict = self.random_noise_1(process_dict)
        process_dict = self.random_jpeg_1(process_dict)
        process_dict = self.random_mpeg_1(process_dict)

        ## the second degradation
        process_dict = self.random_blur_2(process_dict)
        process_dict = self.random_resize_2(process_dict)
        process_dict = self.random_noise_2(process_dict)
        process_dict = self.random_jpeg_2(process_dict)
        process_dict = self.random_mpeg_2(process_dict)

        ## final resize
        process_dict = self.resize_final(process_dict)
        process_dict = self.blur_final(process_dict)


        
        # post process
        process_dict = self.clip(process_dict)
        process_dict = self.rescale.transform(process_dict)



        out_dict['lqs_1'] = process_dict['lqs']
        out_dict['gts_1'] = process_dict['gts']


        ######################################################

        img_gts = deepcopy(img_gts_list_processed)
        img_gts = single_random_crop(img_gts, gt_size)

        num_periods = random.randint(1, 3)
        
        img_gts, selected_2 = self.random_RandomTranslate(img_gts, len(img_gts), num_periods=num_periods)

        cursor = 0
        segments_2 = []
        for idx, length in selected_2:
            if cursor < idx:
                segments_2.append((cursor, idx))          # before the selected region
            segments_2.append((idx, idx + length))        # selected region
            cursor = idx + length

        if cursor < self.num_frame:
            segments_2.append((cursor, self.num_frame))     # remaining tail


        img_lqs = deepcopy(img_gts)

        process_dict = {'lqs': img_lqs, 'gts': img_gts}

        # out_dict = self.usm.transform(out_dict)

        # ## the first degradation
        process_dict = self.random_blur_1(process_dict)
        process_dict = self.random_resize_1(process_dict)
        process_dict = self.random_noise_1(process_dict)
        process_dict = self.random_jpeg_1(process_dict)
        process_dict = self.random_mpeg_1(process_dict)

        ## the second degradation
        process_dict = self.random_blur_2(process_dict)
        process_dict = self.random_resize_2(process_dict)
        process_dict = self.random_noise_2(process_dict)
        process_dict = self.random_jpeg_2(process_dict)
        process_dict = self.random_mpeg_2(process_dict)

        ## final resize
        process_dict = self.resize_final(process_dict)
        process_dict = self.blur_final(process_dict)


        
        # post process
        process_dict = self.clip(process_dict)
        process_dict = self.rescale.transform(process_dict)



        out_dict['lqs_2'] = process_dict['lqs']
        out_dict['gts_2'] = process_dict['gts']


        ######################################################

        img_gts = deepcopy(img_gts_list_processed)
        img_gts = single_random_crop(img_gts, gt_size)

        num_periods = random.randint(1, 3)
        
        img_gts, selected_3 = self.random_RandomTranslate(img_gts, len(img_gts), num_periods=num_periods)

        cursor = 0
        segments_3 = []
        for idx, length in selected_3:
            if cursor < idx:
                segments_3.append((cursor, idx))          # before the selected region
            segments_3.append((idx, idx + length))        # selected region
            cursor = idx + length

        if cursor < self.num_frame:
            segments_3.append((cursor, self.num_frame))     # remaining tail

        img_lqs = deepcopy(img_gts)

        process_dict = {'lqs': img_lqs, 'gts': img_gts}

        # out_dict = self.usm.transform(out_dict)

        # ## the first degradation
        process_dict = self.random_blur_1(process_dict)
        process_dict = self.random_resize_1(process_dict)
        process_dict = self.random_noise_1(process_dict)
        process_dict = self.random_jpeg_1(process_dict)
        process_dict = self.random_mpeg_1(process_dict)

        ## the second degradation
        process_dict = self.random_blur_2(process_dict)
        process_dict = self.random_resize_2(process_dict)
        process_dict = self.random_noise_2(process_dict)
        process_dict = self.random_jpeg_2(process_dict)
        process_dict = self.random_mpeg_2(process_dict)

        ## final resize
        process_dict = self.resize_final(process_dict)
        process_dict = self.blur_final(process_dict)


        
        # post process
        process_dict = self.clip(process_dict)
        process_dict = self.rescale.transform(process_dict)



        out_dict['lqs_3'] = process_dict['lqs']
        out_dict['gts_3'] = process_dict['gts']

        ######################################################

        img_gts = deepcopy(img_gts_list_processed)
        img_gts = single_random_crop(img_gts, gt_size)

        num_periods = random.randint(1, 3)
        
        img_gts, selected_4 = self.random_RandomTranslate(img_gts, len(img_gts), num_periods=num_periods)


        cursor = 0
        segments_4 = []
        for idx, length in selected_4:
            if cursor < idx:
                segments_4.append((cursor, idx))          # before the selected region
            segments_4.append((idx, idx + length))        # selected region
            cursor = idx + length

        if cursor < self.num_frame:
            segments_4.append((cursor, self.num_frame))     # remaining tail

        img_lqs = deepcopy(img_gts)

        process_dict = {'lqs': img_lqs, 'gts': img_gts}

        # out_dict = self.usm.transform(out_dict)

        # ## the first degradation
        process_dict = self.random_blur_1(process_dict)
        process_dict = self.random_resize_1(process_dict)
        process_dict = self.random_noise_1(process_dict)
        process_dict = self.random_jpeg_1(process_dict)
        process_dict = self.random_mpeg_1(process_dict)

        ## the second degradation
        process_dict = self.random_blur_2(process_dict)
        process_dict = self.random_resize_2(process_dict)
        process_dict = self.random_noise_2(process_dict)
        process_dict = self.random_jpeg_2(process_dict)
        process_dict = self.random_mpeg_2(process_dict)

        ## final resize
        process_dict = self.resize_final(process_dict)
        process_dict = self.blur_final(process_dict)


        
        # post process
        process_dict = self.clip(process_dict)
        process_dict = self.rescale.transform(process_dict)



        out_dict['lqs_4'] = process_dict['lqs']
        out_dict['gts_4'] = process_dict['gts']



        for k in out_dict.keys():
            # numpy -> tensor
            out_dict[k] = img2tensor(out_dict[k], bgr2rgb=False)
            out_dict[k] = torch.stack(out_dict[k], dim=0)
            out_dict[k] = out_dict[k].permute(1,0,2,3)
        out_dict['video_shape'] = out_dict['lqs_1'].shape
        out_dict['text'] = init_text
        out_dict['path_1'] = os.path.join(self.save_path+'/Parts_1/', os.path.basename(video_path))
        out_dict['path_2'] = os.path.join(self.save_path+'/Parts_2/', os.path.basename(video_path))
        out_dict['path_3'] = os.path.join(self.save_path+'/Parts_3/', os.path.basename(video_path))
        out_dict['path_4'] = os.path.join(self.save_path+'/Parts_4/', os.path.basename(video_path))


            
        out_dict['segments_1'] = segments_1
        out_dict['segments_2'] = segments_2
        out_dict['segments_3'] = segments_3
        out_dict['segments_4'] = segments_4
        # print(f'selected_1 {segments_1}, selected_2 {segments_2}, selected_3 {segments_3}, selected_4 {segments_4}')
        return out_dict

    def __len__(self):
        return len(self.path)


if __name__ == "__main__":
    # 读取 YAML 配置文件
    config_file = "/gemini/code/cjy/DiffSynth-Studio-main/examples/wanvideo/train_config.yaml"
    with open(config_file, "r") as f:
        opt = yaml.safe_load(f)


    # 设置视频路径
    video_path = "/gemini/code/cjy/High_quality_training_data/Test_Video_for_dataset"  # 修改为实际视频路径

    # 初始化数据集
    dataset = RealVSRRecurrentDataset(opt, video_path, num_frame=49, img_size=(512, 832))

    # 初始化 DataLoader
    batch_size = 1
    num_workers = 4
    shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    save_dir = '/opt/data/private/cjy/data/vae_test_data_multi_test'
    os.makedirs(save_dir, exist_ok=True)
    fps = 30  # 可以根据需要修改
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码

    # 迭代遍历数据集
    for batch_idx, data in enumerate(dataloader):
        lq = data['lqs_']  # (batch_size, num_frames, channels, height, width)
        gt = data['gts_1']  # (batch_size, num_frames, channels, height, width)
        # wo_sharp = data['wo_sharp']
        # 每个 batch 保存
        for b in range(lq.size(0)):
            # 取得视频的分辨率
            _, _, height, width = lq[b].shape

            # 构建保存路径
            gt_video_path = os.path.join(save_dir, f"gt_batch_{batch_idx+1}_sample_{b+1}.mp4")
            gt_writer = cv2.VideoWriter(gt_video_path, fourcc, fps, (width, height))
            # wo_sharp_video_path = os.path.join(save_dir, f"wo_sharp_batch_{batch_idx+1}_sample_{b+1}.mp4")
            # wo_sharp_writer = cv2.VideoWriter(wo_sharp_video_path, fourcc, fps, (width, height))
            lq_video_path = os.path.join(save_dir, f"lq_batch_{batch_idx+1}_sample_{b+1}.mp4")
            lq_writer = cv2.VideoWriter(lq_video_path, fourcc, fps, (width, height))
            # 将每帧写入视频
            for f in range(lq.size(2)):
                # 还原 [-1, 1] 到 [0, 255]
                gt_frame = gt[b, :, f].permute(1, 2, 0).cpu().numpy()
                lq_frame = lq[b, :, f].permute(1, 2, 0).cpu().numpy()
                wo_sharp_frame = wo_sharp[b, :, f].permute(1, 2, 0).cpu().numpy()
                gt_frame = ((gt_frame + 1) * 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
                lq_frame = ((lq_frame + 1) * 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
                wo_sharp_frame = ((wo_sharp_frame + 1) * 127.5).astype(np.uint8)
                # 转换为 BGR 并写入视频
                gt_writer.write(cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))
                lq_writer.write(cv2.cvtColor(lq_frame, cv2.COLOR_RGB2BGR))
                wo_sharp_writer.write(cv2.cvtColor(wo_sharp_frame, cv2.COLOR_RGB2BGR))
            gt_writer.release()
            lq_writer.release()
            wo_sharp_writer.release()
            print(f"Saved {video_path}")

