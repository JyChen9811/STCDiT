import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
import sys
import torch.distributed as dist
sys.path.append('/home/notebook/data/group/cjy/STCDiT-main')
from diffsynth import ModelManager, load_state_dict, save_video
from diffsynth.pipelines.wan_video_t2v_tiny import WanVideoPipeline_t2v_tiny as WanVideoPipeline
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import v2
from glob import glob
import yaml
import random
import time

class TensorDataset(Dataset):
    def __init__(self, image_paths, video_paths):
        self.data_collection = {
            'image': np.array(image_paths),
            'video': np.array(video_paths)
        }
        self.data_lens = {
            'image': len(image_paths),
            'video': len(video_paths)
        }
        self.type_list = ['image', 'video']

    def __getitem__(self, path):
        path, data_type, idx = path
        data = torch.load(path, weights_only=True, map_location="cpu")
        data['dataset_type'] = data_type
        data['path'] = path
        data['idx'] = idx
        
        return data

    def __len__(self):
        return sum(self.data_lens.values())


class val_TensorDataset(torch.utils.data.Dataset):
    def __init__(self, steps_per_epoch):
        self.path = sorted(glob(os.path.join('./validation_pth', '**', '*.pth'), recursive=True))
        print(len(self.path), "validation_data.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        total_steps = 20000,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.denoising_model().enable_lq_condition()
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            training_layer_key_param = ['patch_embedding', 'lq_zero_proj', 'conv', 'key_frame']
            for name, param in self.pipe.denoising_model().named_parameters():
                if any(k in name for k in training_layer_key_param):
                    param.requires_grad = True
                    print(f"Set requires_grad=True for: {name}")

        elif train_architecture == "lora_v2":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            training_layer_key_param = ['patch_embedding', 'head']
            for name, param in self.pipe.denoising_model().named_parameters():
                if any(k in name for k in training_layer_key_param):
                    param.requires_grad = True
                    print(f"Set requires_grad=True for: {name}")

        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.warmup_steps = 500
        self.total_steps = total_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")


    def training_step(self, batch, batch_idx):
        latents = batch["latents"].to(self.device)
        lq_latents = batch["lq_latents"].to(self.device)
        prompt_emb = batch["prompt_emb"].to(self.device)
        image_emb = {}
        key_frame_idx = batch["key_idx_list"].to(self.device)
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents, lq_input = lq_latents, key_frame_idx=key_frame_idx, timestep=timestep, context=prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)



        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


    def validation_step(self, batch, batch_idx):

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
                    fixed[0] = 1
                preprocessed.append(fixed)

            one_counts = [lst.count(1) for lst in preprocessed]
            min_ones = min(one_counts)

            balanced = []
            retained_indices = []

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

        key_frame_idx = batch['video_chunk_idx']
        key_idx = [expand_pattern(key_frame_idx)]
        key_idx_list, retained_indices = balance_ones_with_indices(key_idx)
        
        latents = batch["lq_latents"].to(self.device)
        lq_latents = batch["lq_latents"].to(self.device)
        video_chunk_idx = batch["video_chunk_idx"]
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        
        neg_prompt_emb = batch["neg_prompt_emb"]
        neg_prompt_emb["context"] = neg_prompt_emb["context"][0].to(self.device)
        fake_latents = self.pipe.test(prompt_emb_posi = prompt_emb, prompt_emb_nega=neg_prompt_emb, lq_latents=lq_latents, seed=42, key_frame_idx=retained_indices, num_frames=(lq_latents.shape[2]-1) * 4 + 1, num_inference_steps=20, cfg_scale = 4, device = self.device)
        local_rank = self.trainer.local_rank if hasattr(self.trainer, "local_rank") else 0
        fake_latents_path = os.path.join(
            self.trainer.default_root_dir, f"iter_{self.global_step}_{batch_idx}_rank{local_rank}.pth"
        )

        data = {"fake_latents": fake_latents, 'video_chunk_idx': video_chunk_idx}
        torch.save(data, fake_latents_path)
        loss = torch.nn.functional.mse_loss(fake_latents.float(), latents.float())
        self.log('val_loss', loss)
        return {"val_loss": loss}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=True,
        choices=["train", 'data_decode'],
        help="Task. `train` or `data_decode`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
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
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
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
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=5,
        help="The number of steps between validation checks.",
    )
    parser.add_argument(
        "--checkpoint_every_n_train_steps",
        type=int,
        default=1000,
        help="The number of training steps between checkpoints.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full", 'lora_v2'],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=8,
        help="Number of GPUs.",
    )
    parser.add_argument(
        "--image_batch_size_per_gpu",
        type=int,
        default=0,
        help="Image batch size per GPU.",
    )
    parser.add_argument(
        "--video_batch_size_per_gpu",
        type=int,
        default=4,
        help="Video batch size per GPU.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args

class MultitypeDistributedBatchSampler(Sampler):
    def __init__(self, image_paths, video_paths, batch_size_map, data_prob, num_replicas, rank, seed=0, steps_per_epoch=1000):
        self.image_paths = image_paths
        self.video_paths = video_paths
        self.batch_size_map = batch_size_map
        self.data_prob = data_prob
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch

        self.total_batches = steps_per_epoch
        self.batch_size = 1
        self.image_indices = list(zip(image_paths, ['image'] * len(image_paths), range(len(image_paths))))
        self.video_indices = list(zip(video_paths, ['video'] * len(video_paths), range(len(video_paths))))
        prob_map = dict(zip(['image', 'video'], self.data_prob))
        for data_type, batch_size in self.batch_size_map.items():
            if prob_map[data_type] == 0 and batch_size == 0:
                continue
            if batch_size < self.num_replicas or batch_size % self.num_replicas != 0:
                raise ValueError(
                    f"{data_type} batch size ({batch_size}) must be divisible by "
                    f"num_replicas ({self.num_replicas}) and at least num_replicas."
                )

    def __iter__(self):
        print(f"set {self.seed}")
        rng = np.random.default_rng(self.seed)
        image_pool = rng.permutation(self.image_indices).tolist()
        video_pool = rng.permutation(self.video_indices).tolist()
        
        image_pos = 0
        video_pos = 0

        for _ in range(self.total_batches):
            data_type = rng.choice(['image', 'video'], p=self.data_prob)
            batch_size = self.batch_size_map[data_type]
            per_rank = batch_size // self.num_replicas

            if data_type == 'image':
                if image_pos + batch_size > len(image_pool):
                    remain = image_pool[image_pos:]
                    image_pool = remain + rng.permutation(self.image_indices).tolist()
                    image_pos = 0
                batch = image_pool[image_pos:image_pos + batch_size]
                image_pos += batch_size
            else:
                if video_pos + batch_size > len(video_pool):
                    remain = video_pool[video_pos:]
                    video_pool = remain + rng.permutation(self.video_indices).tolist()
                    
                    video_pos = 0
                    
                batch = video_pool[video_pos:video_pos + batch_size]
                video_pos += batch_size
            
            yield batch[self.rank * per_rank:(self.rank + 1) * per_rank]

    def __len__(self):
        return self.total_batches


def train(args):

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
                fixed[0] = 1
            preprocessed.append(fixed)

        one_counts = [lst.count(1) for lst in preprocessed]
        min_ones = min(one_counts)
        if min_ones >=3:
            min_ones=3
        balanced = []
        retained_indices = []

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
        
    def collate_fn(batch):
        crop_size = 60
        crop_frames = 9

        LQ_list = []
        GT_list = []
        prompt_list = []
        path_list = []
        idx_list = []
        key_idx_list = []
        for example in batch:
            if example["dataset_type"] == 'image':
                _, _, h, w = example["lq_latents"].shape
                top = random.randint(0, max(h - crop_size, 0))
                left = random.randint(0, max(w - crop_size, 0))
                LQ = example["lq_latents"][:, :, top:top+crop_size, left:left+crop_size]
                GT = example["latents"][:, :, top:top+crop_size, left:left+crop_size]
                key_idx = [1]

            else:
                _, T, h, w = example["lq_latents"].shape
                key_idx = expand_pattern(example["video_chunk_idx"])
                t_begin = random.randint(0, T - crop_frames)
                top = random.randint(0, max(h - crop_size, 0))
                left = random.randint(0, max(w - crop_size, 0))
                LQ = example["lq_latents"][:, t_begin:t_begin+crop_frames, top:top+crop_size, left:left+crop_size]
                GT = example["latents"][:, t_begin:t_begin+crop_frames, top:top+crop_size, left:left+crop_size]
                key_idx = key_idx[t_begin:t_begin+crop_frames]

            LQ_list.append(LQ)
            GT_list.append(GT)
            prompt_list.append(example["prompt_emb"]["context"][0])
            path_list.append(example["path"])
            idx_list.append(example['idx'])
            key_idx_list.append(key_idx)
        
        key_idx_list, retained_indices = balance_ones_with_indices(key_idx_list)

        return {
            "lq_latents": torch.stack(LQ_list),
            "latents": torch.stack(GT_list),
            "prompt_emb": torch.stack(prompt_list),
            "path": path_list,
            "idx_list": idx_list,
            "key_idx_list": torch.tensor(retained_indices),
        }

    def train_dataloader(rank):


        image_paths = sorted(glob(os.path.join('./dataset/image/pth_file', '**', '*.pth'), recursive=True))
        video_paths = sorted(glob(os.path.join('./dataset/video/pth_file', '**', '*.pth'), recursive=True))

        dataset = TensorDataset(image_paths, video_paths)
        batch_size_map = {
            'image': args.gpu_num * args.image_batch_size_per_gpu,
            'video': args.gpu_num * args.video_batch_size_per_gpu,
        }

        sampler = MultitypeDistributedBatchSampler(
            image_paths=image_paths,
            video_paths=video_paths,
            batch_size_map=batch_size_map,
            data_prob=[0, 1], #paper use 0, 1
            num_replicas=args.gpu_num,
            rank=rank,
            steps_per_epoch=args.steps_per_epoch,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn
        )

    val_dataset = val_TensorDataset(
        steps_per_epoch=1,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        total_steps=args.steps_per_epoch * args.max_epochs,
        lora_rank=args.lora_rank,
        lora_alpha= args.lora_alpha,
        train_architecture=args.train_architecture,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    tensor_logger = TensorBoardLogger(save_dir=os.path.join(args.output_path, "tensorboardlog"),name="v1")
    logger = [tensor_logger]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=args.checkpoint_every_n_train_steps)],
        logger=logger,
        log_every_n_steps=1,
        val_check_interval = args.val_check_interval,
        use_distributed_sampler=False,
    )
    print(f"local_rank {trainer.local_rank}")
    trainer.fit(model, train_dataloader(trainer.local_rank), val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "train":
        train(args)

