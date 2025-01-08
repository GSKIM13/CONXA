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
        self.head2_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head3_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head4_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head5_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head2 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head3 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head4 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.head5 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4,bias=False), build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU(inplace=True))
        self.fp16_enabled = False

    def forward(self, mla_b2, mla_b3, mla_b4, mla_b5, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)

        head2_1 = self.head2_1(mla_b2)
        head3_1 = self.head3_1(mla_b3)
        head4_1 = self.head4_1(mla_b4)
        head5_1 = self.head5_1(mla_b5)

        return torch.cat([head2, head3, head4, head5, head2_1, head3_1, head4_1, head5_1], dim=1)



@HEADS.register_module()
class VIT_BIMLAHead_CASE8(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_BIMLAHead_CASE8, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.fp16_enabled = False

        self.mlahead_depth = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.global_features_depth = nn.Sequential(
            nn.Conv2d(8 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True))
        self.edge_depth = nn.Conv2d(self.mlahead_channels, 1, 1)


        self.mlahead_normal = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)

        self.global_features_normal = nn.Sequential(
            nn.Conv2d(8 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True))
        self.edge_normal = nn.Conv2d(self.mlahead_channels, 1, 1)

        self.mlahead_ref = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)

        self.global_features_ref = nn.Sequential(
            nn.Conv2d(8 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True))
        self.edge_ref = nn.Conv2d(self.mlahead_channels, 1, 1)

        self.mlahead_illu = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)

        self.global_features_illu = nn.Sequential(
            nn.Conv2d(8 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            build_norm_layer(self.norm_cfg, self.mlahead_channels)[1], nn.ReLU(inplace=True))
        self.edge_illu = nn.Conv2d(self.mlahead_channels, 1, 1)

        
    def forward(self, inputs):
        b6 = inputs[0].chunk(4, dim=1)
        b12 = inputs[1].chunk(4, dim=1)
        b18 = inputs[2].chunk(4, dim=1)
        b24 = inputs[3].chunk(4, dim=1)
        p6 = inputs[4].chunk(4, dim=1)
        p12 = inputs[5].chunk(4, dim=1)
        p18 = inputs[6].chunk(4, dim=1)
        p24 = inputs[7].chunk(4, dim=1)
        #print("here")
        #exit()
        x_depth = self.mlahead_depth(b6[0], b12[0], b18[0], b24[0],p6[0], p12[0], p18[0], p24[0])
        x_depth = self.global_features_depth(x_depth)
        edge_depth = self.edge_depth(x_depth)
        edge_depth = torch.sigmoid(edge_depth)
        
        x_normal = self.mlahead_normal(b6[1], b12[1], b18[1], b24[1],p6[1], p12[1], p18[1], p24[1])
        x_normal = self.global_features_normal(x_normal)
        edge_normal = self.edge_normal(x_normal)
        edge_normal = torch.sigmoid(edge_normal)

        x_ref = self.mlahead_ref(b6[2], b12[2], b18[2], b24[2],p6[2], p12[2], p18[2], p24[2])
        x_ref = self.global_features_ref(x_ref)
        edge_ref = self.edge_ref(x_ref)
        edge_ref = torch.sigmoid(edge_ref)

        x_illu = self.mlahead_illu(b6[3], b12[3], b18[3], b24[3],p6[3], p12[3], p18[3], p24[3])
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
        #print("==")
        return edge, x
