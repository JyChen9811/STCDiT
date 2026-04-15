import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def zero_module(module):
    """Initialize weights to ones and bias to zeros, making the module an identity mapping."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
class MaskUnit(nn.Module):
    ''' 
    Generates the mask and applies the gumbel softmax trick 
    '''

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(MaskUnit, self).__init__()
        self.linear = zero_module(nn.Linear(channels, 1, bias=True))
        self.gumbel = Gumbel()
    def forward(self, x):
        soft_score = self.linear(x) # b n 1

        hard = self.gumbel(soft_score, 1, False) # b n 1
        num_greater_than_0_5 = torch.sum(soft_score == 1).item()
        print(num_greater_than_0_5)
        return hard


    def forward_gumbel_mask(self, lq_input):
        b_lq, c_lq, f_lq, h_lq, w_lq = lq_input.shape
        lq_input = rearrange(lq_input, 'b c f h w -> (b f) c h w',  b = b_lq, f = f_lq).contiguous()
        soft = self.maskconv(lq_input, b_lq, f_lq) # b*f 1 h w
        num_greater_than_0_5 = torch.sum(soft > 0.5).item()
        print(num_greater_than_0_5, soft)
        soft = rearrange(soft, '(b f) c h w -> b f (h w) c',  b = b_lq, f = f_lq).contiguous()
        hard = self.gumbel(soft, 1, False) # b c f (h w)
        print(hard.shape)
        hard = hard.expand(-1, -1, -1, 1) # b*f 1 h w

        hard_dilate = rearrange(hard, 'b f n c -> b (f n) c').contiguous()

        
        return hard_dilate

## Mask convs
class Squeeze(nn.Module):
    """ 
    Squeeze module to predict masks 
    """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride,
                              kernel_size=1, padding=0, bias=True)

    def forward(self, x, b_lq, f_lq):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1) # b 1 1 1
        z = self.conv(x) # b 1 h w

        return z + y.expand_as(z) # b 1 h w



class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-5):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        # Ensure the input is in bfloat16

        if not self.training:  # no Gumbel noise during inference
            return (x > 0).to(dtype=x.dtype)

        if gumbel_noise:
            eps = self.eps
            U1 = torch.rand_like(x, dtype=x.dtype)
            U2 = torch.rand_like(x, dtype=x.dtype)

            # Gumbel noise generation in bfloat16
            g1 = -torch.log(-torch.log(U1 + eps) + eps)
            g2 = -torch.log(-torch.log(U2 + eps) + eps)
            x = x + g1 - g2

        # Sigmoid and straight-through estimator in bfloat16
        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft > 0.5).to(dtype=x.dtype) - soft).detach() + soft
        
        # Sanity check for NaNs
        assert not torch.any(torch.isnan(hard)), "NaNs detected in the output"
        return hard
