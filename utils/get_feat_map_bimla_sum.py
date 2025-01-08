import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import pandas as pd 
import os
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
import numpy as np

from layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.init as init

'''
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

'''

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
        
class Conv_BIMLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256, norm_cfg=None):
        super(Conv_BIMLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_p5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))

        self.mla_b2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_b3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_b4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.mla_b5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                    build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU(inplace=True))
        self.fp16_enabled = False
    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x
    @auto_fp16()
    def forward(self, res2, res3, res4, res5):

        res2 = self.to_2D(res2)  #[4, 1024, 20, 20]
        res3 = self.to_2D(res3)  
        res4 = self.to_2D(res4)  
        res5 = self.to_2D(res5) 

        mla_p5_1x1 = self.mla_p5_1x1(res5) #[4, 256, 20, 20]
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1) #[4, 256, 20, 20]
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        mla_b2_plus = mla_p2_1x1
        mla_b3_plus = mla_b2_plus + mla_p3_1x1
        mla_b4_plus = mla_b3_plus + mla_p4_1x1
        mla_b5_plus = mla_b4_plus + mla_p5_1x1

        mla_b2 = self.mla_b2(mla_b2_plus)
        mla_b3 = self.mla_b3(mla_b3_plus)
        mla_b4 = self.mla_b4(mla_b4_plus)
        mla_b5 = self.mla_b5(mla_b5_plus)

        return mla_b2, mla_b3, mla_b4, mla_b5, mla_p2, mla_p3, mla_p4, mla_p5
        
                       
        
class ConvNeXt_V2_CASE2_LN(nn.Module):
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
    def __init__(self, in_chans=3, model_name = 'convnext_large_224', num_classes=1000, mla_channels=192, mla_index = (0,1,2,3),
                 depths=[3, 3, 27, 3], dims=[192, 384, 768,1536], drop_path_rate=0., drop_rate=0., embed_dim = 192,
                 layer_scale_init_value=1e-6, head_init_scale=1., norm_cfg=dict(type='BN', requires_grad=True), norm_layer=partial(nn.LayerNorm, eps=1e-6), category_emb_dim = 128,
                 scale = 64
                 ):
        super().__init__()
        
        self.model_name = model_name
        
        self.embed_dim = embed_dim
        self.mla_channels = mla_channels
        self.mla_index = mla_index
        self.norm_cfg = norm_cfg
        self.random_init = False
        
        self.scale = scale
        
        self.attention_dim = 4*self.embed_dim

        self.norm_layer = norm_layer
        
        self.category_emb_dim = category_emb_dim

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
        
        self.deconv_1 = nn.ConvTranspose2d(dims[1], self.embed_dim, kernel_size=2, stride =2)
        self.deconv_2 = nn.ConvTranspose2d(dims[2], self.embed_dim, kernel_size=4, stride =4)
        self.deconv_3 = nn.ConvTranspose2d(dims[3], self.embed_dim, kernel_size=8, stride =8)
        
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(dims[1], self.embed_dim, kernel_size=2, stride =2), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(dims[2], self.embed_dim, kernel_size=4, stride =4), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(dims[3], self.embed_dim, kernel_size=8, stride =8), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True), nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding =1), nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True))       
        
        self.mla_depth = Conv_BIMLA(in_channels=self.embed_dim, mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)
        self.mla_normal = Conv_BIMLA(in_channels=self.embed_dim, mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)
        self.mla_ref = Conv_BIMLA(in_channels=self.embed_dim, mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)
        self.mla_illu = Conv_BIMLA(in_channels=self.embed_dim, mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)

        
        self.norm_0 = norm_layer(self.embed_dim)
        self.norm_1 = norm_layer(self.embed_dim)
        self.norm_2 = norm_layer(self.embed_dim)
        self.norm_3 = norm_layer(self.embed_dim)  
        
        #self.apply(self._init_weights)
        


      
        #self.category_embedding = nn.Parameter(torch.randn(6400, self.category_emb_dim), requires_grad =  True)
        

        self.fp16_enabled = False
        
    def init_weights_category(self):
        init.kaiming_normal_(self.category_embedding, mode = 'fan_in', nonlinearity='relu')
        
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
        


        out = []
        
        out.append(outs[0])       
        out.append(outs[0]+outs[1])
        out.append(outs[0]+outs[1]+outs[2])
        out.append(outs[0]+outs[1]+outs[2]+outs[3])
        out.append(outs[0]+outs[1]+outs[2]+outs[3])
        out.append(outs[1]+outs[2]+outs[3])
        out.append(outs[2]+outs[3])
        out.append(outs[3])

    
        
         
        return out


def cosine_similarity(tensor_a, tensor_b):
    return torch.nn.functional.cosine_similarity(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0))
          
        
import torch

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd 
from PIL import Image
from torchvision import transforms

file_path = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/case3_bimla_add_conv/iter_4000.pth'



file_name = file_path.split('/')[-2]
iter_name = file_path.split('/')[-1][:-4]
state_dict = torch.load(file_path)

keys_to_delete = [key for key in state_dict['state_dict'].keys() if key.startswith('decode_head')]

for key in keys_to_delete:
    del state_dict['state_dict'][key]
    
model = ConvNeXt_V2_CASE2_LN()

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items():
    new_key = k[9:]
    new_state_dict[new_key] = v

#
state_dict['state_dict'] = new_state_dict




model.load_state_dict(state_dict['state_dict'], strict=False)

model.eval()


img_dir = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/test/'

img_list = os.listdir(img_dir)

dict_ = {0 : 'depth', 1 : 'normal', 2 : 'ref', 3: 'illu'}

img_list = img_list[:2]

for img in img_list:
  image_path = os.path.join(img_dir, img)
  image = Image.open(image_path).convert('RGB')
  
  
  transform = transforms.Compose([
      transforms.CenterCrop((320, 320)),
      transforms.ToTensor()
  ])
  
  input_tensor = transform(image).unsqueeze(0)


  with torch.no_grad(): 
    output = model(input_tensor)
    
  for i in range(len(output)): 
    result = output[i]
    result = result.squeeze(0)
    for j in range(len(result)):

  
      plt.figure(figsize=(6, 6))
      ax = sns.heatmap(result[j], cmap='Greys_r', fmt=".3f",cbar=False)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlabel('')
      ax.set_ylabel('')
      os.makedirs(f'/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/bimla_feat_map_new_sum/', exist_ok = True)
      plt.savefig(f'/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/bimla_feat_map_new_sum/{img}_{i}_{j}.png')
      plt.show()
      plt.close()
