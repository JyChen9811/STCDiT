import types
from ..models import ModelManager
from ..models.wan_video_dit_i2v_large import WanModel_i2v_large
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from peft import LoraConfig, inject_adapter_in_model
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt

class LocalAttention3D:
    def __init__(self, kernel_size=(3, 128, 128), overlap=(0.5, 0.5, 0.5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.overlap = overlap
        
    def grids(self, x):
        b, c, f, h, w = x.shape
        self.original_size = (b, c, f, h, w)
        kf, kh, kw = self.kernel_size
        
        # 防止kernel超出边界
        
        kf = min(kf, f)
        kh = min(kh, h)
        kw = min(kw, w)

        self.tile_weights = self._gaussian_weights(kf, kh, kw)

        # 计算步长
        step_f = kf if f == kf else max(1, int(kf * self.overlap[0]))
        step_h = kh if h == kh else max(1, int(kh * self.overlap[1]))
        step_w = kw if w == kw else max(1, int(kw * self.overlap[2]))
        
        parts = []
        idxes = []

        for fi in range(0, f, step_f):
            if fi + kf > f:
                fi = f - kf
            for hi in range(0, h, step_h):
                if hi + kh > h:
                    hi = h - kh
                for wi in range(0, w, step_w):
                    if wi + kw > w:
                        wi = w - kw
                    parts.append(x[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw])

                    idxes.append({'f': fi, 'h': hi, 'w': wi})

        self.idxes = idxes
        return torch.cat(parts, dim=0)


    def grids_lq(self, x, key_frame_list):
        b, c, f, h, w = x.shape
        self.original_size = (b, c, f, h, w)
        kf, kh, kw = self.kernel_size
        
        # 防止kernel超出边界
        
        kf = min(kf, f)
        kh = min(kh, h)
        kw = min(kw, w)

        self.tile_weights = self._gaussian_weights(kf, kh, kw)

        # 计算步长
        step_f = kf if f == kf else max(1, int(kf * self.overlap[0]))
        step_h = kh if h == kh else max(1, int(kh * self.overlap[1]))
        step_w = kw if w == kw else max(1, int(kw * self.overlap[2]))
        
        parts = []
        idxes = []
        key_frame_idx = []

        for fi in range(0, f, step_f):
            if fi + kf > f:
                fi = f - kf
            for hi in range(0, h, step_h):
                if hi + kh > h:
                    hi = h - kh
                for wi in range(0, w, step_w):
                    if wi + kw > w:
                        wi = w - kw
                    parts.append(x[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw])

                    idxes.append({'f': fi, 'h': hi, 'w': wi})
                    key_frame_idx.append([key_frame_list[0][fi:fi + kf]])
        self.idxes = idxes
        return torch.cat(parts, dim=0), key_frame_idx

    def _gaussian_weights(self, tile_depth, tile_height, tile_width):
        var = 0.01
        midpoint_d = (tile_depth - 1) / 2
        midpoint_h = (tile_height - 1) / 2
        midpoint_w = (tile_width - 1) / 2
        
        # 计算各个维度上的权重
        d_probs = [exp(-(d-midpoint_d)*(d-midpoint_d)/(tile_depth*tile_depth)/(2*var)) / sqrt(2*pi*var) for d in range(tile_depth)]
        h_probs = [exp(-(h-midpoint_h)*(h-midpoint_h)/(tile_height*tile_height)/(2*var)) / sqrt(2*pi*var) for h in range(tile_height)]
        w_probs = [exp(-(w-midpoint_w)*(w-midpoint_w)/(tile_width*tile_width)/(2*var)) / sqrt(2*pi*var) for w in range(tile_width)]

        # 生成3D高斯权重
        weights = np.outer(np.outer(d_probs, h_probs).reshape(-1), w_probs).reshape(tile_depth, tile_height, tile_width)
        return torch.tensor(weights, device=torch.device('cuda')).unsqueeze(0).repeat(16, 1, 1, 1)

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, f, h, w = self.original_size
        count_mt = torch.zeros((b, 16, f, h, w)).to(outs.device)
        kf, kh, kw = self.kernel_size

        for cnt, each_idx in enumerate(self.idxes):
            fi = each_idx['f']
            hi = each_idx['h']
            wi = each_idx['w']
            preds[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw] += outs[cnt, :, :, :, :] * self.tile_weights
            count_mt[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw] += self.tile_weights

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        qkv = self.grids(x)
        out = self.grids_inverse(qkv)
        return out




class WanVideoPipeline_i2v_large(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel_i2v_large = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()




    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit_lq_i2v_large")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline_i2v_large(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def denoising_model_load_lora(self, lora_config):
        self.dit = inject_adapter_in_model(lora_config, self.dit)
        return self.dit

    def denoising_model_enable_lq_input(self):
        return self.dit.enable_lq_condition()

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}



    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_lq = None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.frame_process = v2.Compose([
            v2.Resize(size=(height, width), antialias=True),
        ])
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)


        self.load_models_to_device(['vae'])
        # self.vae.to(device=noise.device)
        input_lq = self.preprocess_images(input_lq)
        input_lq = torch.stack(input_lq, dim=2).to(dtype=self.torch_dtype, device=self.device)
        input_lq = self.frame_process(input_lq)
        input_lq_latents = self.encode_video(input_lq, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        latents = self.scheduler.add_noise(input_lq_latents, noise, timestep=self.scheduler.timesteps[0])


        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        # self.text_encoder.to(device=noise.device)
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(self.dit, latents, lq_input= input_lq_latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi, **usp_kwargs)
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(self.dit, latents, lq_input= input_lq_latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, **usp_kwargs)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])
        self.vae.to(device=noise.device)
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames

    def pad_input_lq(self, input_lq: torch.Tensor):
        B, C, F, H, W = input_lq.shape

        pad_frames = 0

        if F % 4 != 1:
            target_frames = ((F + 2) // 4) * 4 + 1
            pad_frames = target_frames - F

            # 取最后一帧 (shape: B, C, 1, H, W)
            last_frame = input_lq[:, :, -1:, :, :]  # 保留frame维度

            # 扩展重复最后一帧 pad_frames 次
            padding = last_frame.expand(-1, -1, pad_frames, -1, -1).clone()
            input_lq = torch.cat([input_lq, padding], dim=2)  # concat along frame dim
        B, C, F, H, W = input_lq.shape

        return input_lq, pad_frames

    def encode_and_padding_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):

        input_video, pad_frames = self.pad_input_lq(input_video)

        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents



    @torch.no_grad()
    def test_tlc_seg(
        self,
        prompt,
        negative_prompt="",
        segments = None,
        input_lq = None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        tile_kernel=None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        temporal_overlap=0.95,
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check

        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.frame_process = v2.Compose([
            v2.Resize(size=(height, width), antialias=True),
        ])
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)



        self.load_models_to_device(['vae'])

        input_lq = self.preprocess_images(input_lq)
        input_lq = torch.stack(input_lq, dim=2).to(dtype=self.torch_dtype, device=self.device)
        input_lq = self.frame_process(input_lq)
        pre_idx = 0
        lq_latents_list = []
        video_chunk_idx = []
        input_lq_frame_count = []

        print(f"segments {segments}")
        

        for start, end in segments:
            lq_latents_part = self.encode_and_padding_video(input_lq[:, :, start:end, :, :], **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)[0]

            lq_latents_list.append(lq_latents_part)
            
            video_chunk_idx.append(lq_latents_part.shape[1])
            input_lq_frame_count.append(input_lq[:, :, start:end, :, :].shape[2])

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

            return retained_indices


        key_frame_idx = [expand_pattern(video_chunk_idx)]
        
        
        input_lq_latents = torch.concat(lq_latents_list, dim=1).unsqueeze(0)

        noise = self.generate_noise((1, 16, input_lq_latents.shape[2], height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        latents = self.scheduler.add_noise(input_lq_latents, noise, timestep=self.scheduler.timesteps[0])

        if input_lq_latents.shape[2] < 21:
            tile_frame_kernel, height_kernel, width_kernel = tile_kernel
            tile_kernel = (input_lq_latents.shape[2], height_kernel, width_kernel)

        self.tlc_vae_latents = LocalAttention3D(tile_kernel, (temporal_overlap, 0.95, 0.95))
        self.tlc_vae_img = LocalAttention3D(tile_kernel, (temporal_overlap, 0.95, 0.95))


        latents = self.tlc_vae_latents.grids(latents)
        input_lq_latents, key_frame_idx = self.tlc_vae_img.grids_lq(input_lq_latents, key_frame_idx)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        # self.text_encoder.to(device=noise.device)
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()
        sub_latents_num = latents.shape[0]
        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if progress_id >=1 :
                latents = self.tlc_vae_latents.grids(latents).to(dtype=latents.dtype)
            concat_grid = []
            # Inference

            for sub_num in range(sub_latents_num):
                
                sub_latents = latents[sub_num, :, :, :, :].unsqueeze(0)
                input_lq_sub_latents = input_lq_latents[sub_num, :, :, :, :].unsqueeze(0)
                key_frame_sub_idx = key_frame_idx[sub_num]
                
                retained_sub_indices = balance_ones_with_indices(key_frame_sub_idx)

                noise_pred_posi = model_fn_wan_video_tlc_test(self.dit, sub_latents, lq_input= input_lq_sub_latents, key_frame_idx=retained_sub_indices, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi, **usp_kwargs)
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video_tlc_test(self.dit, sub_latents, lq_input= input_lq_sub_latents, key_frame_idx=retained_sub_indices, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, **usp_kwargs)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                sub_latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], sub_latents)
                concat_grid.append(sub_latents)
            
            latents = self.tlc_vae_latents.grids_inverse(torch.cat(concat_grid, dim=0)).to(sub_latents.dtype)
        # Decode
        self.load_models_to_device(['vae'])
        self.vae.to(device=noise.device)
        
        total_frames = []
        pre_idx = 0 
        for frame_count, frame_num in zip(input_lq_frame_count, video_chunk_idx):

            frames = self.decode_video(latents[:, :,pre_idx:pre_idx+frame_num, :, :], **tiler_kwargs)
            pre_idx = pre_idx+frame_num
            frames = self.tensor2video(frames[0])
            total_frames += frames[:frame_count]
        self.load_models_to_device([])
        return total_frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel_i2v_large, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video_tlc_test(
    dit: WanModel_i2v_large,
    x: torch.Tensor,
    lq_input: torch.Tensor,
    key_frame_idx: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    **kwargs,
):
    se1 = sinusoidal_embedding_1d(dit.freq_dim, timestep).to(timestep.device)

    t = dit.time_embedding(se1)
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    
    y = lq_input
    b, c, f, h ,w = lq_input.shape
    msk = torch.ones((b, 4, f, h, w), device=lq_input.device, dtype=lq_input.dtype)
    y = torch.concat([msk, y],dim=1)
    x_input = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
    
    x, (f, h, w) = dit.patchify(x_input)
    frames = key_frame_idx[0][::4]  # 每 4 帧anchor取一帧
    if frames[-1] != key_frame_idx[0][-1]:  
        frames.append(key_frame_idx[0][-1]) 
    key_frame_idx = [frames]

    key_frame_feat, retained_indices, key_frame_token_lens, key_f = dit.extract_key_frame_patch_embedding(lq_input, key_frame_idx, f, h, w)

    x = torch.concat([x, key_frame_feat], dim=1)

    shape = (key_f, f, h, w)

    freqs = torch.cat([
        dit.freqs[0][:f+key_f].view(f+key_f, 1, 1, -1).expand(f+key_f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f+key_f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f+key_f, h, w, -1)
    ], dim=-1).reshape((f+key_f) * h * w, 1, -1).to(x.device)
    

    for idx, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs, shape, retained_indices)

    x, _ = torch.split(x, [x.shape[1]-key_frame_token_lens, key_frame_token_lens], dim=1)

    x = dit.head(x, t)

    x = dit.unpatchify(x, (f, h, w))

    return x








