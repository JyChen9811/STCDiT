# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange


# class MaskUnit(nn.Module):
#     ''' 
#     Generates the mask and applies the gumbel softmax trick 
#     '''

#     def __init__(self, channels, stride=1, dilate_stride=1):
#         super(MaskUnit, self).__init__()
#         self.maskconv = Squeeze(channels=channels, stride=stride)
        
#     def forward(self, lq_input):
#         b_lq, c_lq, f_lq, h_lq, w_lq = lq_input.shape
#         lq_input = rearrange(lq_input, 'b c f h w -> (b f) c h w',  b = b_lq, f = f_lq).contiguous()
#         soft = self.maskconv(lq_input, b_lq, f_lq) # b*f 1 h w
#         soft = rearrange(soft, '(b f) c h w -> b c f (h w)',  b = b_lq, f = f_lq).contiguous()

#         hard = F.gumbel_softmax(soft, dim = -1, hard = True) # b c f (h w)
#         hard = hard.expand(-1, c_lq, -1, -1) # b*f 1 h w

#         hard_dilate = rearrange(hard, 'b c f n -> b (f n) c').contiguous()

        
#         return hard_dilate


#     def forward_gumbel_mask(self, lq_input):
#         b_lq, c_lq, f_lq, h_lq, w_lq = lq_input.shape
#         lq_input = rearrange(lq_input, 'b c f h w -> (b f) c h w',  b = b_lq, f = f_lq).contiguous()
#         soft = self.maskconv(lq_input, b_lq, f_lq) # b*f 1 h w
#         num_greater_than_0_5 = torch.sum(soft > 0.5).item()
#         print(num_greater_than_0_5, soft)
#         soft = rearrange(soft, '(b f) c h w -> b f (h w) c',  b = b_lq, f = f_lq).contiguous()
#         hard = F.gumbel_softmax(abs(soft), dim = -1, hard = True) # b c f (h w)
#         print(hard.shape)
#         # hard = hard.expand(-1, 1, -1, -1) # b*f 1 h w

#         hard_dilate = rearrange(hard, 'b f n c -> b (f n) c').contiguous()

        
#         return hard_dilate

# ## Mask convs
# class Squeeze(nn.Module):
#     """ 
#     Squeeze module to predict masks 
#     """

#     def __init__(self, channels, stride=1):
#         super(Squeeze, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(channels, 1, bias=True)
#         self.conv = nn.Conv2d(channels, 1, stride=stride,
#                               kernel_size=1, padding=0, bias=True)

#     def forward(self, x, b_lq, f_lq):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, 1, 1, 1) # b 1 1 1
#         z = self.conv(x) # b 1 h w

#         return z + y.expand_as(z) # b 1 h w




