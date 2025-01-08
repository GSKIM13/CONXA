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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.fp16_enabled = False

    @auto_fp16()
    def forward(self, x):
        B, C, H, W = x.shape  #(1,3,512,512)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        #print(type(x))
        #exit()
        x = self.proj(x) #1,1024,32,32
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.fp16_enabled = False

    @auto_fp16()
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

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
        
               
        
class ViT_Cross_Att(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, model_name='vit_large_patch16_384', img_size=320, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                 num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None, 
                 pos_embed_interp=False, random_init=False, align_corners=False, mla_channels=256, category_emb_dim = 128, embed_dim_1 = 64,
                 scale = 64,
                 mla_index=(5,11,17,23), **kwargs):
        super(ViT_Cross_Att, self).__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.mla_channels = mla_channels
        self.mla_index = mla_index
        self.category_emb_dim = category_emb_dim
        self.embed_dim_1 = embed_dim_1
        self.attention_dim = embed_dim_1*4
        
        self.scale = scale

        self.num_stages = self.depth
        self.out_indices= tuple(range(self.num_stages))

        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))   #[1, 1, 1024]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))  #[1, 1025, 1024]
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])      
            
        self.deconv_0 = nn.ConvTranspose2d(1024, self.embed_dim_1, kernel_size=4, stride =4)
        self.deconv_1 = nn.ConvTranspose2d(1024, self.embed_dim_1, kernel_size=4, stride =4)
        self.deconv_2 = nn.ConvTranspose2d(1024, self.embed_dim_1, kernel_size=4, stride =4)
        self.deconv_3 = nn.ConvTranspose2d(1024, self.embed_dim_1, kernel_size=4, stride =4)
        
    
        self.query_li = nn.Linear(self.category_emb_dim, self.category_emb_dim) ## from feature
        self.key_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.value_li = nn.Linear(self.attention_dim, self.attention_dim) ## from embedding
        self.softmax = nn.Softmax(dim=-1)
        
        self.category_embedding = nn.Parameter(torch.empty(6400, self.category_emb_dim), requires_grad=True)
        
        
        self.query_norm = nn.LayerNorm(self.category_emb_dim)
        self.key_norm = nn.LayerNorm(self.attention_dim)
        self.value_norm = nn.LayerNorm(self.attention_dim)        

        
        self.layer_norm = LayerNorm(self.category_emb_dim, eps=1e-6, data_format="channels_first")

        self.fp16_enabled = False
        


    @auto_fp16()
    def forward(self, x):
        
        #print("here!!")
        B = x.shape[0] #[4, 3, 320, 320]
        x = self.patch_embed(x) #[4, 1024, 20, 20]
        x = x.flatten(2).transpose(1, 2) #[4, 400, 1024]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks [1, 1, 1024]
        x = torch.cat((cls_tokens, x), dim=1)  #[4, 401, 1024]
        x = x + self.pos_embed  #[4, 401, 1024]
        x = x[:,1:]   #[4, 400, 1024]
        x = self.pos_drop(x) 
        outs = []  #len(outs) = 24
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
                
        B,HW,D = outs[self.mla_index[0]].shape
        
        import math
        
        H = W = int(math.sqrt(HW))

        #c6 = self.norm_0(outs[self.mla_index[0]])   #[4, 400, 1024] mla_index=(5, 11, 17, 23) torch.Size([8, 400, 1024])        
        #c12 = self.norm_1(outs[self.mla_index[1]])  #[4, 400, 1024]
        #c18 = self.norm_2(outs[self.mla_index[2]])  #[4, 400, 1024]
        #c24 = self.norm_3(outs[self.mla_index[3]])  #[4, 400, 1024]
        
        c6 = outs[self.mla_index[0]].view(B,H,W,D).permute(0,3,1,2) # 4,1024,20,20
        c12 = outs[self.mla_index[1]].view(B,H,W,D).permute(0,3,1,2)
        c18 = outs[self.mla_index[2]].view(B,H,W,D).permute(0,3,1,2)
        c24 = outs[self.mla_index[3]].view(B,H,W,D).permute(0,3,1,2)
               
        
        c6 = self.deconv_0(c6)
        c12 = self.deconv_0(c12)
        c18 = self.deconv_0(c18)
        c24 = self.deconv_0(c24)
        
        batch_size, _, height, width = c6.shape
        
        outs_concat = torch.cat([c6,c12,c18,c24],dim=1).flatten(2).permute(0,2,1) #4,4096,20,20 -> 4,4096,400 -> 4,400,4096
        
        
        #self.category_embedding : (6400,256)
              
        category_embed = self.category_embedding.expand(batch_size, -1,-1) #torch.Size([2, 400, 256]) # embedding 
        
        query = self.query_li(category_embed).permute(0,2,1) # query : torch.Size([4, 256, 6400])

        key = self.key_li(outs_concat)  #key : torch.Size([4, 6400, 1024])


        value = self.value_li(outs_concat).permute(0,2,1) #feature
        
        query = self.query_norm(query.permute(0, 2, 1)).permute(0, 2, 1)  # Back to original shape
        key = self.key_norm(key)
        value = self.value_norm(value.permute(0, 2, 1)).permute(0, 2, 1)
            
        energy = torch.bmm(query, key) 
        attention = self.softmax(energy / (self.scale ** 0.5)) 
        
        out = torch.bmm(attention, value)  # out : torch.Size([4, 256, 6400])
        
               
        out = out.view(batch_size, self.category_emb_dim, height, width) # torch.Size([4, 256, 80, 80])

        
        out = self.layer_norm(out)
        

        
        
        
         
        return out, attention
        
        
import torch

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd 
from PIL import Image
from torchvision import transforms

file_path = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/Vit_Cross_Att_rev2/iter_4000.pth'



file_name = file_path.split('/')[-2]
iter_name = file_path.split('/')[-1][:-4]
state_dict = torch.load(file_path)

keys_to_delete = [key for key in state_dict['state_dict'].keys() if key.startswith('decode_head')]

for key in keys_to_delete:
    del state_dict['state_dict'][key]
    
model = ViT_Cross_Att()

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items():
    new_key = k[9:]
    new_state_dict[new_key] = v

#
state_dict['state_dict'] = new_state_dict

print(state_dict['state_dict']['category_embedding'])
#
#print("Keys in state_dict['state_dict'] with 'backbone.':", state_dict['state_dict'].keys())


model.load_state_dict(state_dict['state_dict'], strict=False)

model.eval()


img_dir = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/test/'

img_list = os.listdir(img_dir)

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

  _, rows, cols = output[1].shape



  result = output[1].squeeze(0)
  
  
  block_size_row = rows // 4
  block_size_col = cols // 4
  
  
  resampled_data = np.zeros((4, 4))
  
  for i in range(4):
      for j in range(4):
          block = result[i*block_size_row:(i+1)*block_size_row, j*block_size_col:(j+1)*block_size_col]
          resampled_data[i, j] = block.sum()/64
  
  
  plt.figure(figsize=(8, 6))
  ax = sns.heatmap(resampled_data, cmap='Greys_r', annot = True, fmt=".3f", vmin=0.2, vmax=0.3)
  plt.title('Heatmap of Summed Data (4x4)')
  os.makedirs(f'/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/{file_name}_{iter_name}', exist_ok = True)
  plt.savefig(f'/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/{file_name}_{iter_name}/{img}')
  plt.show()
  plt.close()
  
