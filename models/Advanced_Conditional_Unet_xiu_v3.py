import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .Advanced_Network_Helpers import *
from transformers import PreTrainedModel
from CLIP.CLIP_relation import *
from process_v2 import DynamicLineAndStyleFusion
from Swin_Encoder.net.Swin_Encoder import Swin_Encoder
from torchvision import transforms
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        config11=None,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels  # since we are concatenating the images and the conditionings along the channel dimension

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(self.channels * 2, init_dim, 7, padding=3)
        self.conditioning_init = nn.Conv2d(self.channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.in_out = in_out

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.conditioning_encoder = nn.ModuleList([])
        num_resolutions = len(in_out)
        self.num_resolutions = num_resolutions
        self.Swin_Encoder=Swin_Encoder(config11)
        self.DynamicLineAndStyleFusion=DynamicLineAndStyleFusion(
        style_dim=320, 
        line_dim=512, 
        d_model=256, 
        num_heads=8
        )
        # conditioning encoder
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.conditioning_encoder.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block1_ = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        # self.LinearAttention_xiu=LinearAttention_xiu(256)
        self.cross_attention_1 = Residual(
            PreNorm(mid_dim, LinearCrossAttention_xiu())
        )
        self.cross_attention_2 = Residual(
            PreNorm(mid_dim, LinearCrossAttention_xiu())
        )
        self.cross_attention_3 = Residual(
            PreNorm(mid_dim, LinearCrossAttention(mid_dim))
        )
        self.attention_1 = Residual(
            PreNorm(mid_dim, LinearAttention(mid_dim))
        )
        self.attention_2 = Residual(
            PreNorm(mid_dim, LinearAttention(mid_dim))
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        self.mid_block2_ = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim),nn.GroupNorm(32, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, implicit_conditioning, explicit_conditioning,model_clip,current_step,total_step):
        x = torch.cat((x, implicit_conditioning), dim=1)

        x = self.init_conv(x)


        preprocess = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # 从 [-1, 1] 到 [0, 1]
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 然后在需要的地方应用这个管道
        implicit_conditioning = preprocess(implicit_conditioning)
        
        lineart_feature=CLIP_image(implicit_conditioning,model_clip)#线稿部分的融合
       
        style_feature=self.Swin_Encoder(explicit_conditioning)




        conditioning=self.DynamicLineAndStyleFusion(style_feature,lineart_feature,current_step,total_step)

        
       
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        

        # conditioning encoder
        # conditioning = self.conditioning_init(implicit_conditioning)
        # for block1, attn, downsample in self.conditioning_encoder:
        #     conditioning = block1(conditioning)
        #     conditioning = attn(conditioning)
        #     conditioning = downsample(conditioning)


#unet的下采样
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            # x = attn(x)
            h.append(x)
            x = downsample(x)
        # print(x.shape)
        
        # reverse the c list

        # bottleneck

        x = self.cross_attention_1(x, conditioning)
       
        x = self.mid_block1(x, t)
        x = self.attention_1(x)
        x = self.mid_block1_(x,t)
      
        x = self.cross_attention_2(x, conditioning)
        x = self.attention_2(x)
        x = self.mid_block2(x, t)


###unet的上采样
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        return self.final_conv(x)
