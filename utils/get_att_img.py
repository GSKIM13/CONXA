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

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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
    def __init__(self, in_chans=3, model_name = 'convnext_large_224', num_classes=1000, mla_channels=256, mla_index = (0,1,2,3),
                 depths=[3, 3, 27, 3], dims=[192, 384, 768,1536], drop_path_rate=0., drop_rate=0., embed_dim = 32,
                 layer_scale_init_value=1e-6, head_init_scale=1., norm_cfg = None, norm_layer=partial(nn.LayerNorm, eps=1e-6), category_emb_dim = 256,
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
        
        self.apply(self._init_weights)


        

        
        self.conv_0 = nn.Conv2d(dims[0], self.embed_dim, kernel_size =1 )
        
        self.deconv_1 = nn.ConvTranspose2d(dims[1], self.embed_dim, kernel_size=2, stride =2)
        self.deconv_2 = nn.ConvTranspose2d(dims[2], self.embed_dim, kernel_size=4, stride =4)
        self.deconv_3 = nn.ConvTranspose2d(dims[3], self.embed_dim, kernel_size=8, stride =8)
        
        self.query_li = nn.Linear(self.category_emb_dim, self.category_emb_dim) ## from feature
        self.key_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.value_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.softmax = nn.Softmax(dim=-1)
        
        self.category_embedding = nn.Parameter(torch.full((6400, self.category_emb_dim), 0.5), requires_grad=True)
        
        self.init_weights_category()
        
        #self.category_embedding = nn.Parameter(torch.randn(6400, self.category_emb_dim), requires_grad =  True)
        
        self.query_norm = nn.LayerNorm(self.category_emb_dim)
        self.key_norm = nn.LayerNorm(self.attention_dim)
        self.value_norm = nn.LayerNorm(self.attention_dim)
        
        self.layer_norm = LayerNorm(self.category_emb_dim, eps=1e-6, data_format="channels_first")

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
            

                      
           
        outs[0] = self.conv_0(outs[0])
            
        outs[1] = self.deconv_1(outs[1])
        outs[2] = self.deconv_2(outs[2])
        outs[3] = self.deconv_3(outs[3]) # [2, 32, 80, 80]
        
        
        batch_size, _, height, width =outs[0].shape
        
        outs_concat = torch.cat([outs[0],outs[1],outs[2],outs[3]],dim=1).flatten(2).permute(0,2,1) # 2,128,80,80 -> 2, 6400, 128

        
        #self.category_embedding : (6400,4)
        
        
              
        category_embed = self.category_embedding.expand(batch_size, -1,-1) #torch.Size([2, 6400, 256]) # embedding 
        
        query = self.query_li(category_embed).permute(0,2,1) #torch.Size([2, 6400, 256]) -> (2,256,6400) # embedding
        
        key = self.key_li(outs_concat)  #feature 2,6400,128
        
        value = self.value_li(outs_concat).permute(0,2,1) #feature 2,128,6400
        
        query = self.query_norm(query.permute(0, 2, 1)).permute(0, 2, 1)  # Back to original shape
        key = self.key_norm(key)
        value = self.value_norm(value.permute(0, 2, 1)).permute(0, 2, 1) # 1*128*6400

        
              
        
        energy = torch.bmm(query, key)  # (2,256,128)
        attention = self.softmax(energy / (self.scale ** 0.5))  # (2,256,128) 
        
        piece_result = []
        
        fea_piece = self.attention_dim // 4 #64
        cat_piece =  self.category_emb_dim // 4 #32
        
        
        for cat in range(4):
          for fea in range(4):
            result = torch.bmm(attention[:,cat*cat_piece:(cat+1)*cat_piece,fea*fea_piece:(fea+1)*fea_piece], value[:,fea*fea_piece:(fea+1)*fea_piece,:])
            result= result.reshape(1,64,80,80)
            piece_result.append(result)

                       
        
        out = torch.bmm(attention, value)  # [B, HW, D] (2,256,6400)
        


                     
        out = out.view(batch_size, self.category_emb_dim, height, width) # (2,256,80,80)
        
        out = self.layer_norm(out)
        
        
        
         
        return out, attention, piece_result
        
        
import torch

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd 
from PIL import Image
from torchvision import transforms

file_path = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/5000_100000_iter3/iter_2000.pth'



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

name_list = ['depth_1','depth_2','depth_3','depth_4','illu_1','illu_2','illu_3','illu_4','normal_1','normal_2','normal_3','normal_4','ref_1','ref_2','ref_3','ref_4']

for img in img_list:
  image_path = os.path.join(img_dir, img)
  image = Image.open(image_path).convert('RGB')
  
  
  transform = transforms.Compose([
      transforms.CenterCrop((320, 320)),
      transforms.ToTensor()
  ])
  
  input_tensor = transform(image).unsqueeze(0)


  with torch.no_grad(): 
    output = model(input_tensor)[2]

  for i in range(16):
    result_tensor = output[i]
    result_tensor=result_tensor.mean(axis=1)
    
    to_pil = transforms.ToPILImage()
    image = to_pil(result_tensor)
    
    #img_folder = f'/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/0_500000_scale64_iter3/img'
    
    img_folder = os.path.join(file_path[:-14], 'img')
    
    os.makedirs(img_folder, exist_ok=True)
    

    
    img_dir_ = os.path.join(img_folder, img[:-4]+ '_' + name_list[i] + '.jpg')
    
    image.save(img_dir_)
    
  
  

  
