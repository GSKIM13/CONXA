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


class BIMLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=64, norm_cfg=None):
        super(BIMLAHead, self).__init__()
        #self.head2_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
        #                             nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        #self.head3_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
        #                             nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        #self.head4_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
        #                             nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        #self.head5_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
        #                             nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        self.head2 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        self.head3 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        self.head4 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        self.head5 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(inplace=True))
        self.fp16_enabled = False

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)

        #head2_1 = self.head2_1(mla_b2)
        #head3_1 = self.head3_1(mla_b3)
        #head4_1 = self.head4_1(mla_b4)
        #head5_1 = self.head5_1(mla_b5)

        return torch.cat([head2, head3, head4, head5], dim=1)



@HEADS.register_module()
class VIT_BIMLAHead_CASE7_MLA(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_BIMLAHead_CASE7_MLA, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.fp16_enabled = False

        self.mlahead_depth = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.global_features_depth = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True))
        self.edge_depth = nn.Conv2d(self.mlahead_channels, 1, 1)

        
        self.mlahead_normal = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
    
        self.global_features_normal = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True))
        self.edge_normal = nn.Conv2d(self.mlahead_channels, 1, 1)
        
        self.mlahead_ref = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_ref = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True))
        self.edge_ref = nn.Conv2d(self.mlahead_channels, 1, 1)
        
        self.mlahead_illu = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        
        self.global_features_illu = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(inplace=True))
        self.edge_illu = nn.Conv2d(self.mlahead_channels, 1, 1)
        
    def forward(self, inputs):
        inputs_ = inputs[0].chunk(4, dim=1)
        depth = inputs_[0].chunk(4, dim=1)
        normal = inputs_[1].chunk(4, dim=1)
        ref = inputs_[2].chunk(4, dim=1)
        illu = inputs_[3].chunk(4, dim=1)
        x_depth = self.mlahead_depth(depth[0], depth[1], depth[2], depth[3])
        x_depth = self.global_features_depth(x_depth)
        edge_depth = self.edge_depth(x_depth)
        edge_depth = torch.sigmoid(edge_depth)
        
        x_normal = self.mlahead_normal(normal[0], normal[1], normal[2], normal[3])
        x_normal = self.global_features_normal(x_normal)
        edge_normal = self.edge_normal(x_normal)
        edge_normal = torch.sigmoid(edge_normal)

        x_ref = self.mlahead_ref(ref[0], ref[1], ref[2], ref[3])
        x_ref = self.global_features_ref(x_ref)
        edge_ref = self.edge_ref(x_ref)
        edge_ref = torch.sigmoid(edge_ref)

        x_illu = self.mlahead_illu(illu[0], illu[1], illu[2], illu[3])
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
