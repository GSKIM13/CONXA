import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer


class BIMLAHead_BN_ConvNeXt(nn.Module):
    def __init__(self, in_channel = 32, middle_channels=32, head_channels=64, norm_cfg=None):
        super(BIMLAHead_BN_ConvNeXt, self).__init__()
        
        self.in_channel = in_channel     
        self.middle_channels = middle_channels
        self.head_channels = head_channels
        
           
        self.head = nn.Sequential(nn.ConvTranspose2d(self.in_channel, self.middle_channels, 2, stride=2,  bias=False), build_norm_layer(norm_cfg, self.middle_channels)[1], nn.ReLU(inplace=True),  nn.ConvTranspose2d(self.middle_channels, self.head_channels, 2, stride=2, bias=False),build_norm_layer(norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.fp16_enabled = False

    def forward(self, x):
        x = self.head(x)
        return x



@HEADS.register_module()
class CONXA_Head(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, middle_channels=256, head_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, category_emb_dim = 32, **kwargs):
        super(CONXA_Head, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.middle_channels = middle_channels
        self.BatchNorm = norm_layer
        self.middle_channels = middle_channels
        self.head_channels = head_channels
        self.in_channel = category_emb_dim // 4

        self.fp16_enabled = False

        self.mlahead_depth = BIMLAHead_BN_ConvNeXt(in_channel = self.in_channel, middle_channels = self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_depth = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.edge_depth = nn.Conv2d(self.head_channels, 1, 1)

        
        self.mlahead_normal = BIMLAHead_BN_ConvNeXt(in_channel = self.in_channel, middle_channels = self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_normal = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.edge_normal = nn.Conv2d(self.head_channels, 1, 1)
        
        self.mlahead_ref = BIMLAHead_BN_ConvNeXt(in_channel = self.in_channel, middle_channels = self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_ref = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.edge_ref = nn.Conv2d(self.head_channels, 1, 1)

        
        self.mlahead_illu = BIMLAHead_BN_ConvNeXt(in_channel = self.in_channel, middle_channels = self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_illu = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.edge_illu = nn.Conv2d(self.head_channels, 1, 1)
        
    def forward(self, inputs):
    
        inputs = inputs.chunk(4, dim = 1)

        
        depth = inputs[0] # 1,80,80
        normal = inputs[1]
        ref = inputs[2]
        illu = inputs[3]
        

        x_depth = self.mlahead_depth(depth)

        x_depth = self.global_features_depth(x_depth)

        edge_depth = self.edge_depth(x_depth)
        edge_depth = torch.sigmoid(edge_depth)
        
        x_normal = self.mlahead_normal(normal)
        x_normal = self.global_features_normal(x_normal)
        edge_normal = self.edge_normal(x_normal)
        edge_normal = torch.sigmoid(edge_normal)

        x_ref = self.mlahead_ref(ref)
        x_ref = self.global_features_ref(x_ref)
        edge_ref = self.edge_ref(x_ref)
        edge_ref = torch.sigmoid(edge_ref)

        x_illu = self.mlahead_illu(illu)
        x_illu = self.global_features_illu(x_illu)
        edge_illu = self.edge_illu(x_illu)
        edge_illu = torch.sigmoid(edge_illu)

        x = torch.cat([x_depth,x_normal,x_ref,x_illu], dim=1)
        #x = self.global_features_depth(x)
        #edge = self.edge_depth(x)
        #edge = torch.sigmoid(edge)
        
        edge = torch.cat([edge_depth,edge_normal,edge_ref,edge_illu], dim=1)
        
        #x = torch.cat([x_depth,x_normal,x_ref], dim=1)
        #edge = torch.cat([edge_depth,edge_normal,edge_ref], dim=1)

        return edge, x
