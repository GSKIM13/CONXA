###########################################################################
# Created by: pmy
# Copyright (c) 2019
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.vit import Block

from mmcv.cnn import build_norm_layer


class SFTLayer(nn.Module):
    def __init__(self, head_channels):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(head_channels, head_channels, 1)

        self.SFT_shift_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(head_channels, head_channels, 1)

    def forward(self, local_features, global_features):
        #print('=====local_features=====global_features=====')
        #print(local_features.shape, global_features.shape)
        scale = self.SFT_scale_conv1(F.relu(self.SFT_scale_conv0(global_features),inplace=True))
        shift = self.SFT_shift_conv1(F.relu(self.SFT_shift_conv0(global_features),inplace=True))
        fuse_features = local_features * (scale+1) +shift
        return fuse_features

@HEADS.register_module()
class Local8x8_fuse_head(BaseDecodeHead):

    #def __init__(self, img_size=320, mla_channels=128, mlahead_channels=64,
    def __init__(self, img_size=320, mla_channels=128, mlahead_channels=64,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(Local8x8_fuse_head, self).__init__(**kwargs)

        self.img_size = img_size
        self.channels = mla_channels
        self.head_channels = mlahead_channels
        self.norm_cfg = norm_cfg
        self.BatchNorm = norm_layer
        
        self.class_num = 1

        self.SFT_head_depth = SFTLayer(self.head_channels*self.class_num)
        self.SFT_head_normal = SFTLayer(self.head_channels*self.class_num)
        self.SFT_head_ref = SFTLayer(self.head_channels*self.class_num)
        self.SFT_head_illu = SFTLayer(self.head_channels*self.class_num)
        
        self.edge_head_depth = nn.Sequential(
            nn.Conv2d(self.head_channels*self.class_num, self.head_channels*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels*self.class_num), nn.ReLU(),
            nn.Conv2d(self.head_channels*self.class_num, 64*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(64*self.class_num), nn.ReLU(),
            nn.Conv2d(64*self.class_num, 1*self.class_num, 1)
        )

        self.edge_head_normal = nn.Sequential(
            nn.Conv2d(self.head_channels*self.class_num, self.head_channels*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels*self.class_num), nn.ReLU(),
            nn.Conv2d(self.head_channels*self.class_num, 64*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(64*self.class_num), nn.ReLU(),
            nn.Conv2d(64*self.class_num, 1*self.class_num, 1)
        )
        
        self.edge_head_ref = nn.Sequential(
            nn.Conv2d(self.head_channels*self.class_num, self.head_channels*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels*self.class_num), nn.ReLU(),
            nn.Conv2d(self.head_channels*self.class_num, 64*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(64*self.class_num), nn.ReLU(),
            nn.Conv2d(64*self.class_num, 1*self.class_num, 1)
        )

        self.edge_head_illu = nn.Sequential(
            nn.Conv2d(self.head_channels*self.class_num, self.head_channels*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels*self.class_num), nn.ReLU(),
            nn.Conv2d(self.head_channels*self.class_num, 64*self.class_num, 3, padding=1, bias=False),
            nn.BatchNorm2d(64*self.class_num), nn.ReLU(),
            nn.Conv2d(64*self.class_num, 1*self.class_num, 1)
        )
        
        
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local_features, global_features):
    

  
        local_features = torch.chunk(local_features, chunks=4, dim = 1)
        global_features = torch.chunk(global_features, chunks=4, dim = 1)
        
        print(f'local_feature : {local_features[0].size()}')
        print(f'global_feature : {global_features[0].size()}')

        fuse_features_depth = self.SFT_head_depth(local_features[0], global_features[0])      
        fuse_edge_depth = self.edge_head_depth(fuse_features_depth)
        fuse_edge_depth = torch.sigmoid(fuse_edge_depth)
        
        fuse_features_normal = self.SFT_head_normal(local_features[1], global_features[1])
        fuse_edge_normal = self.edge_head_normal(fuse_features_normal)
        fuse_edge_normal = torch.sigmoid(fuse_edge_normal)
        
        fuse_features_ref = self.SFT_head_ref(local_features[2], global_features[2])
        fuse_edge_ref = self.edge_head_normal(fuse_features_ref)
        fuse_edge_ref = torch.sigmoid(fuse_edge_ref)
        
        fuse_features_illu = self.SFT_head_illu(local_features[3], global_features[3])
        fuse_edge_illu = self.edge_head_illu(fuse_features_illu)
        fuse_edge_illu = torch.sigmoid(fuse_edge_illu)       
        
        fuse_edge = torch.cat([fuse_edge_depth, fuse_edge_normal, fuse_edge_ref, fuse_edge_illu], axis =1 )
        fuse_features = torch.cat([fuse_features_depth, fuse_features_normal, fuse_features_ref, fuse_features_illu], axis =1 )
        

        
        fuse_edge = fuse_edge.permute(0,2,3,1)
        
        
        return fuse_edge, fuse_features
