import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import pandas as pd 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import pandas as pd 
import os

from .helpers import load_pretrained_convnext
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES

from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16

def get_unique_filename(base_path, base_name, extension):
    counter = 1
    file_path = f"{base_path}/{base_name}.{extension}"
    while os.path.exists(file_path):
        file_path = f"{base_path}/{base_name}_{counter}.{extension}"
        counter += 1
    return file_path

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth',
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/jx_vit_base_p16_384-83fb41ba.pth'),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/jx_vit_large_p16_384-b3be5167.pth'),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'convnext_large_224': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'convnext_large_384': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
        
    'convnext_v2_large_224': _cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    
    
    
        
        
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
    'deit_base_distilled_path16_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        pretrained_finetune='pretrain/deit_base_distilled_patch16_384.pth'
    )
}

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
        
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
#from MinkowskiEngine import SparseTensor

class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
        
               
        
@BACKBONES.register_module()
class ConvNeXt_V2_CASE8_BN(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, model_name = 'convnext_large_224', num_classes=1000, mla_channels=256, mla_index = (0,1,2,3),
                 depths=[3, 3, 27, 3], dims=[192, 384, 768,1536], drop_path_rate=0., drop_rate=0., embed_dim = 32,
                 layer_scale_init_value=1e-6, head_init_scale=1., norm_cfg = None, norm_layer=partial(nn.LayerNorm, eps=1e-6), category_emb_dim = 128,
                 scale = 128):
        super().__init__()
        
        self.model_name = model_name
        
        self.embed_dim = embed_dim
        self.mla_channels = mla_channels
        self.mla_index = mla_index
        self.norm_cfg = norm_cfg
        self.random_init = False
        
        self.attention_dim = 4*self.embed_dim

        self.norm_layer = norm_layer
        
        self.category_emb_dim = category_emb_dim
        
        self.scale = scale

        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        self.apply(self._init_weights)
        


         
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(dims[1], self.embed_dim, kernel_size=2, stride =2), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(dims[2], self.embed_dim, kernel_size=4, stride =4), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(dims[3], self.embed_dim, kernel_size=8, stride =8), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))
        
        self.query_li = nn.Linear(self.category_emb_dim, self.category_emb_dim) ## from feature
        self.key_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.value_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.softmax = nn.Softmax(dim=-1)
        
        
        self.category_embedding = nn.Parameter(torch.empty(6400, self.category_emb_dim), requires_grad=True)
        
        self.init_weights_category()
        
        
        self.query_norm = nn.BatchNorm1d(self.category_emb_dim)
        self.key_norm = nn.BatchNorm1d(self.attention_dim)
        self.value_norm = nn.BatchNorm1d(self.attention_dim)
        
        self.batch_norm = nn.BatchNorm2d(self.category_emb_dim)

        self.fp16_enabled = False
        
    def init_weights_category(self):
        init.kaiming_normal_(self.category_embedding, mode = 'fan_in', nonlinearity='relu')
        
    def init_weights(self, pretrained=None):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.random_init == False:
            self.default_cfg = default_cfgs[self.model_name]

            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained_convnext(self)
            else:
                load_pretrained_convnext(self)
        else:
            print('Initialize weight randomly')

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
    
        batch_size = len(x)
        outs = []
        
        # x = F.interpolate(x, size =(224,224), mode = 'bilinear', align_corners = False)
        for i in range(len(self.mla_index)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
            outs.append(x)
                                             
        outs[1] = self.deconv_1(outs[1])
        outs[2] = self.deconv_2(outs[2])
        outs[3] = self.deconv_3(outs[3]) # [2, 32, 80, 80]
        
        
        batch_size, _, height, width =outs[0].shape
        
        outs_concat = torch.cat([outs[0],outs[1],outs[2],outs[3]],dim=1).flatten(2).permute(0,2,1) # -> N, 6400, 768
         
        category_embed = self.category_embedding.expand(batch_size, -1,-1) #torch.Size([2, 6400, 256]) # embedding 
        
        query = self.query_li(category_embed) #torch.Size([2, 6400, 512])
        key = self.key_li(outs_concat) #feature 2,6400,768
        value = self.value_li(outs_concat) # 2,6400,768
        
        query = self.query_norm(query.permute(0,2,1)) # 512, 6400
        key = self.key_norm(key.permute(0,2,1)).permute(0,2,1)
        value = self.value_norm(value.permute(0,2,1))
  
        
        energy = torch.bmm(query, key)  # (N,512,768)
        attention = self.softmax(energy / (self.scale**(0.5)))  # (2,512,768)
        
        out = torch.bmm(attention, value)  # [B, HW, D] (2,4,6400)

                     
        out = out.view(batch_size, self.category_emb_dim, height, width) # (2,4,80,80)
        
        out = self.batch_norm(out)
        
        
        
         
        return out
        







