import argparse

import torch
import lightning as pl
from PIL import Image
import yaml
import sys
sys.path.append('/home/notebook/data/group/cjy/STCDiT-main')
from dataset.realbasicvsr_multi_DA_video_dataset import Zoom_translate_RealVSRRecurrentDataset as TextVideoDataset
from diffsynth import ModelManager
from diffsynth.pipelines.wan_video_i2v_large import WanVideoPipeline_i2v_large as WanVideoPipeline
import os
class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
       
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video_shape = batch["text"][0], batch["video_shape"]
        path = None
        try:
            # print(batch["video"])
            self.pipe.device = self.device
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            neg_prompt_emb = self.pipe.encode_prompt('')

            for part_idx in range(1, 5):
                segments = batch[f"segments_{part_idx}"]
                video = batch[f"gts_{part_idx}"]
                lq_video = batch[f"lqs_{part_idx}"]
                path = batch[f"path_{part_idx}"][0]

                if not isinstance(video, torch.Tensor):
                    continue

                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                lq_video = lq_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

                if isinstance(segments, torch.Tensor):
                    segments = segments.squeeze(0).tolist()

                latents_list = []
                lq_latents_list = []
                video_chunk_idx = []
                input_lq_frame_count = []
                for start, end in segments:
                    start = int(start.item() if isinstance(start, torch.Tensor) else start)
                    end = int(end.item() if isinstance(end, torch.Tensor) else end)

                    latents_part = self.pipe.encode_and_padding_video(video[:, :, start:end, :, :], **self.tiler_kwargs).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)[0]
                    lq_latents_part = self.pipe.encode_and_padding_video(lq_video[:, :, start:end, :, :], **self.tiler_kwargs).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)[0]

                    latents_list.append(latents_part)
                    lq_latents_list.append(lq_latents_part)
                    video_chunk_idx.append(lq_latents_part.shape[1])
                    input_lq_frame_count.append(lq_video[:, :, start:end, :, :].shape[2])

                latents = torch.concat(latents_list, dim=1)
                lq_latents = torch.concat(lq_latents_list, dim=1)

                # image
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                data = {"video_chunk_idx": video_chunk_idx, "video_shape": video_shape, "latents": latents, "lq_latents": lq_latents,"prompt_emb": prompt_emb, "neg_prompt_emb": neg_prompt_emb, "image_emb": image_emb}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # print(path, text)
                torch.save(data, path + ".tensors.pth")
                # os.remove(path)
        except:
            print(path)
            # os.remove(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Wan video training tensors.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path for Lightning outputs.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./",
        help="Path for saving tensors.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="./",
        help="Path for video.",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="./",
        help="Path for caption.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    args = parser.parse_args()
    return args


def data_process(args):

    config_file = "./train_config.yaml"
    with open(config_file, "r") as f:
        opt = yaml.safe_load(f)

    video_path = args.video_path
    json_path = args.caption_path
    save_path = args.save_path
    os.makedirs(save_path+'/Parts_1/', exist_ok=True)
    os.makedirs(save_path+'/Parts_2/', exist_ok=True)
    os.makedirs(save_path+'/Parts_3/', exist_ok=True)
    os.makedirs(save_path+'/Parts_4/', exist_ok=True)
    dataset = TextVideoDataset(opt, video_path, json_path, save_path, num_frame=49, img_size=(512, 512), interval_list=[1,2,3,4])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    

if __name__ == '__main__':
    args = parse_args()
    data_process(args)
