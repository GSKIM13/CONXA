import torch
import torch.nn as nn
import torch.nn.functional as F


from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class VIT_MLAHead_SETR(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead_SETR, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = 4

        self.mlahead_depth = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_depth = nn.Conv2d(4 * self.mlahead_channels,1, 3, padding=1)

        self.mlahead_normal = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_normal = nn.Conv2d(4 * self.mlahead_channels,1, 3, padding=1)
        
        self.mlahead_ref = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_ref = nn.Conv2d(4 * self.mlahead_channels,1, 3, padding=1)
        
        self.mlahead_illu = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_illu = nn.Conv2d(4 * self.mlahead_channels,1, 3, padding=1)

    def forward(self, inputs):
        x_depth = self.mlahead_depth(inputs[0], inputs[1], inputs[2], inputs[3])
        x_depth = self.cls_depth(x_depth)
        x_depth = F.interpolate(x_depth, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        x_depth = torch.sigmoid(x_depth)
        
        x_normal = self.mlahead_normal(inputs[0], inputs[1], inputs[2], inputs[3])
        x_normal = self.cls_normal(x_normal)
        x_normal = F.interpolate(x_normal, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        x_normal = torch.sigmoid(x_normal)


        x_ref = self.mlahead_ref(inputs[0], inputs[1], inputs[2], inputs[3])
        x_ref = self.cls_ref(x_ref)
        x_ref = F.interpolate(x_ref, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        x_ref = torch.sigmoid(x_ref)
        
        x_illu = self.mlahead_illu(inputs[0], inputs[1], inputs[2], inputs[3])
        x_illu = self.cls_illu(x_illu)
        x_illu = F.interpolate(x_illu, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        x_illu = torch.sigmoid(x_illu)
        x = torch.cat([x_depth,x_normal,x_ref,x_illu], dim=1)
        return x, x
