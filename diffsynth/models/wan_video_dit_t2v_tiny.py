import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module    

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class key_CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.linear_q(x))
        k = self.norm_k(self.linear_k(ctx))
        v = self.linear_v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.linear_o(x)

# class Key_frame_process_module(nn.Module):
#     def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
#         super().__init__()

#         self.key_frame_cross_attn = key_CrossAttention(
#             dim, num_heads, eps, has_image_input=False)

#         self.key_frame_ffn = nn.Sequential(
#             nn.Conv2d(in_channels=dim, out_channels=2 * dim, kernel_size=1, stride=1),
#             nn.Conv2d(in_channels=2*dim, out_channels=2*dim, kernel_size=3, stride=1, padding=1, groups=2 * dim),
#             nn.GELU(),
#             nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, stride=1),
#         )

#         self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)


#     def forward(self, x, key_frame, shape):
#         h, w = shape
#         enhanced_x = self.key_frame_cross_attn(self.norm1(x), key_frame)
#         x = x + enhanced_x
#         x_input = self.norm2(x)
#         x_input = rearrange(x_input, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
#         x_input = self.key_frame_ffn(x_input)
#         x_input = rearrange(x_input, 'b c h w -> b (h w) c', h=h, w=w).contiguous()
#         x = x + x_input

#         return x


class Key_frame_process_module(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.proj_in_FTC = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_dw_FTC = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.weights_FTC = nn.Parameter(torch.ones(2))
        self.eps = eps

    def forward(self,  x, key_frame, shape):
        h, w = shape
        p_size = (x.shape[2], x.shape[3])
        key_frame_input_down = F.adaptive_max_pool2d(key_frame, p_size)

        relu_weights_FTC = F.relu(self.weights_FTC)
        weight_sum_FTC = relu_weights_FTC.sum() + self.eps
        normalized_weights_FTC = relu_weights_FTC / weight_sum_FTC
        # print(key_frame_input_down.shape, x.shape)
        out_FTC_feat = sum(w * inp for w, inp in zip(normalized_weights_FTC, [key_frame_input_down, x]))

        out_FTC_feat = F.silu(out_FTC_feat)
        out_FTC_feat = self.proj_dw_FTC(self.proj_in_FTC(out_FTC_feat))
        # out_FTC_feat = rearrange(out_FTC_feat, '(b f) c h w -> b c f h w ', f = f).contiguous()
        x = x + out_FTC_feat
        return x




class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()


    def enable_lq_proj(self):

        self.dw_conv = zero_module(nn.Conv3d(in_channels=self.ffn_dim, out_channels=self.ffn_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=self.ffn_dim))
        self.lq_conv = zero_module(nn.Conv3d(in_channels=self.ffn_dim, out_channels=self.ffn_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=self.ffn_dim))
        self.lq_inject_conv = zero_module(nn.Conv3d(in_channels=self.ffn_dim, out_channels=self.ffn_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=self.ffn_dim))
        self.lq_act = nn.GELU()
    def forward(self, x, context, t_mod, freqs, shape, key_frame_idx):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        f_key, f_lq, h_lq, w_lq= shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        x, x_lq = torch.split(x, [f_lq*h_lq*w_lq, x.shape[1]-f_lq*h_lq*w_lq], dim=1)

        x = x + self.cross_attn(self.norm3(x), context)

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_lq = modulate(self.norm2(x_lq), shift_mlp, scale_mlp)
        
        for i, layer in enumerate(self.ffn):
            if i == 0:
                input_x = layer(input_x)
                input_lq = layer(input_lq)

                input_lq_reshape = rearrange(input_lq, 'b (f h w) c -> b c f h w', f=f_key, h=h_lq, w=w_lq).contiguous()
                lq_conv = self.lq_conv(input_lq_reshape)
                input_lq_reshape = input_lq_reshape + lq_conv

                lq_injection = self.lq_inject_conv(input_lq_reshape)
                lq_injection = self.lq_act(lq_injection)
                lq_injection = input_lq_reshape * lq_injection
                lq_injection = rearrange(lq_injection, 'b c f h w -> b (f h w) c').contiguous()
                input_lq = rearrange(input_lq_reshape, 'b c f h w -> b (f h w) c').contiguous()
                
                input_x = self.key_frame_lq_inject(input_x, lq_injection, shape, key_frame_idx)


                x_conv = rearrange(input_x, 'b (f h w) c -> b c f h w', f=f_lq, h=h_lq, w=w_lq).contiguous()
                x_conv = self.dw_conv(x_conv)
                x_conv = rearrange(x_conv, 'b c f h w -> b (f h w) c').contiguous()
                input_x = input_x + x_conv


                   # 相加融合
            else:
                input_x = layer(input_x)
                input_lq = layer(input_lq)

        ffn_x = input_x
        x = self.gate(x, gate_mlp, ffn_x)

        ffn_lq = input_lq
        x_lq = self.gate(x_lq, gate_mlp, ffn_lq)
        ffn_x = torch.concat([x, x_lq], dim=1)
        
        return ffn_x


    def key_frame_lq_inject(self, x, lq, shape, retained_indices):
        B = x.shape[0]
        _, _, H, W = shape
        tokens_per_frame = H * W
        for b in range(B):
            retained = retained_indices[b]
            for idx, key_idx in enumerate(retained):
                x[b, key_idx*H*W : (key_idx+1)*H*W, :] = x[b, key_idx*H*W : (key_idx+1)*H*W, :] + lq[b, idx*H*W : (idx+1)*H*W, :]
                

        return x



class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x




class efficient_align(nn.Module):
    def __init__(self, input_dim, inner_dim):
        super(efficient_align, self).__init__()
        self.inner_dim = inner_dim
        # nn.Linear should output a tensor that can be reshaped to a 4D tensor for nn.Conv2d
        self.proj_in = nn.Linear(input_dim * 2, self.inner_dim)
        self.dw_conv = nn.Conv3d(self.inner_dim, self.inner_dim, kernel_size=(1,3,3), padding=(0,1,1), groups=self.inner_dim)
        self.proj_out = zero_module(nn.Linear(self.inner_dim, input_dim))

    def forward(self, x, frame, height, width):
        # print(x.shape)
        x = self.proj_in(x)
        # print(x.shape, height, width)
        x = rearrange(x, 'b (f h w) c -> b c f h w', f=frame, h=height, w=width)

        x = self.dw_conv(x)

        x = F.gelu(x)

        x = rearrange(x, 'b c f h w -> b (f h w) c')

        x = self.proj_out(x)

        

        return x





class WanModel_t2v_tiny(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb


    def lq_patchify(self, x: torch.Tensor):
        x = self.lq_patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        x = self.lq_zero_proj(x)
        return x, grid_size  # x, grid_size: (f, h, w)


    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def enable_lq_condition(self):
        # 创建一个新的 16 通道卷积（结构必须相同，除了 out_channels）

        for i, block in enumerate(self.blocks):
            # 创建新的LQ_DiTBlock实例
            block.enable_lq_proj()

        # self.align = efficient_align(self.dim, self.dim)


        with torch.no_grad():
            # 初始化 key_frame_patch_embedding
            self.lq_key_frame_patch_embedding = nn.Conv3d(
                in_channels=16,
                out_channels=self.dim,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            )

            # 拷贝 patch_embedding 的前16通道权重
            self.lq_key_frame_patch_embedding.weight.copy_(
                self.patch_embedding.weight
            )

            self.lq_key_frame_patch_embedding.bias.copy_(
                self.patch_embedding.bias
            )


            self.lq_patch_embedding = nn.Conv3d(
                16, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
            # 使用 copy_() 复制权重
            # 使用 clone() 来避免 in-place 操作错误
            self.lq_patch_embedding.weight.data = self.patch_embedding.weight.data.clone()

            # 如果有偏置，也可以类似地复制
            if self.patch_embedding.bias is not None:
                self.lq_patch_embedding.bias.data = self.patch_embedding.bias.data.clone()

            self.lq_zero_proj = zero_module(nn.Linear(self.dim, self.dim))



        self.lq_key_frame_patch_embedding_finescale = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=self.dim, kernel_size=1, stride=1),
            nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1,3,3), stride=1, padding=(0,1,1), groups=self.dim)
        )

        self.lq_key_frame_process = zero_module(Key_frame_process_module(self.dim))



    def extract_key_frame_patch_embedding(self, lq_input, retained_indices, F, H, W):
        """
        Retained frames processed in (B * L, H*W, C) format.

        Args:
            x: (B, F*H*W, C)
            retained_indices: list of list of ints (frame idx per batch)
            H, W: spatial dimensions
            process_fn: function from (B*L, H*W, C) → (B*L, H*W, C)

        Returns:
            x_out: (B, F*H*W, C)
        """
        B = lq_input.shape[0]
        tokens_per_frame = H * W

        key_frame_all = []
        for b in range(B):
            frames_wo_patch = list(torch.split(lq_input[b], 1, dim=1)) # C F H W
            retained = retained_indices[b]
            key_frames = [frames_wo_patch[i] for i in retained] # (C F H W)
            key_frames = torch.cat(key_frames, dim=1).unsqueeze(0)
            key_frame_all.append(key_frames)

        # === Step 1: Process selected frames as (B*L, H*W, C)
        key_frame_selected = torch.cat(key_frame_all, dim=0) # (B C F H W)
        

        key_frame_feat_fine_scale = self.lq_key_frame_patch_embedding_finescale(key_frame_selected)
        _, _, key_f, key_h, key_w = key_frame_feat_fine_scale.shape

        key_frame_feat_fine_scale = rearrange(key_frame_feat_fine_scale, 'b c f h w -> (b f) c h w').contiguous()# (B * L, H*W , C)
        

        key_frame_feat = self.lq_key_frame_patch_embedding(key_frame_selected)

        _, _, key_f, key_h, key_w = key_frame_feat.shape

        key_frame_feat = rearrange(key_frame_feat, 'b c f h w -> (b f) c h w').contiguous()# (B * L, H*W , C)
        
        
        # print(f'key_frame_feat size 1{key_frame_feat.shape}')
        key_frame_feat = self.lq_key_frame_process(key_frame_feat, key_frame_feat_fine_scale, (key_h,key_w))
        # print(f'key_frame_feat size 2{key_frame_feat.shape}')
        key_frame_feat = rearrange(key_frame_feat, '(b f) c h w -> b (f h w) c', f=key_f).contiguous()# (B * L, H*W , C)
        token_lens = key_frame_feat.shape[1]
        return key_frame_feat, retained_indices, token_lens, key_f  # (B, F*H*W, C)


    def forward(self,
                x: torch.Tensor,
                lq_input: torch.Tensor,
                key_frame_idx: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        

        x, (f, h, w) = self.patchify(x)
        
        lq_input_cues, (f, h, w) = self.lq_patchify(lq_input)

        # lq_input_cues = self.align(torch.concat([x, lq_input_cues], dim=2), f, h, w)

        x = x + lq_input_cues


        key_frame_feat, retained_indices, key_frame_token_lens, key_f = self.extract_key_frame_patch_embedding(lq_input, key_frame_idx, f, h, w)



        x = torch.concat([x, key_frame_feat], dim=1)

        shape = (key_f, f, h, w)
        freqs = torch.cat([
            self.freqs[0][:f+key_f].view(f+key_f, 1, 1, -1).expand(f+key_f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f+key_f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f+key_f, h, w, -1)
        ], dim=-1).reshape((f+key_f) * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, shape, retained_indices, 
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, shape, retained_indices,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, shape, retained_indices)

        x, _ = torch.split(x, [x.shape[1]-key_frame_token_lens, key_frame_token_lens], dim=1)
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        else:
            config = {}
        return state_dict, config
