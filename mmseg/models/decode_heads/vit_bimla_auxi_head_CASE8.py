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


@HEADS.register_module()
class VIT_BIMLA_AUXIHead_CASE8(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, **kwargs):
        super(VIT_BIMLA_AUXIHead_CASE8, self).__init__(**kwargs)
        self.img_size = img_size
        #if self.in_channels==1024:
        #    self.aux_0 = nn.Conv2d(self.in_channels, 256, kernel_size=1, bias=False)
        #    self.aux_1 = nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False)
        if self.in_channels==1024:
            self.div_channels = 256
            self.aux_depth = nn.Sequential(
                nn.ConvTranspose2d(self.div_channels, self.div_channels, 4, stride=2, padding=1, bias=False),
                ###nn.ConvTranspose2d(self.in_channels, 1, 16, stride=8, padding=4, bias=False),
                nn.ConvTranspose2d(self.div_channels, 1, 16, stride=8, padding=4, bias=False)
                #nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=False)
            )
            self.aux_normal = nn.Sequential(
                nn.ConvTranspose2d(self.div_channels, self.div_channels, 4, stride=2, padding=1, bias=False),
                ###nn.ConvTranspose2d(self.in_channels, 1, 16, stride=8, padding=4, bias=False),
                nn.ConvTranspose2d(self.div_channels, 1, 16, stride=8, padding=4, bias=False)
                #nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=False)
            )
            self.aux_ref = nn.Sequential(
                nn.ConvTranspose2d(self.div_channels, self.div_channels, 4, stride=2, padding=1, bias=False),
                ###nn.ConvTranspose2d(self.in_channels, 1, 16, stride=8, padding=4, bias=False),
                nn.ConvTranspose2d(self.div_channels, 1, 16, stride=8, padding=4, bias=False)
                #nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=False)
            )
            self.aux_illu = nn.Sequential(
                nn.ConvTranspose2d(self.div_channels, self.div_channels, 4, stride=2, padding=1, bias=False),
                ###nn.ConvTranspose2d(self.in_channels, 1, 16, stride=8, padding=4, bias=False),
                nn.ConvTranspose2d(self.div_channels, 1, 16, stride=8, padding=4, bias=False)
                #nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=False)
            )

    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        x = self._transform_inputs(x)
        inputs_depth, inputs_normal, inputs_ref, inputs_illu = x.chunk(4, dim=1)
        #print(inputs_depth.size())
        if x.dim()==3:
            x = x[:,1:]
            x = self.to_2D(x)
        if self.in_channels==x.shape[1]:
            x_depth = self.aux_depth(inputs_depth)
            x_normal = self.aux_normal(inputs_normal)
            x_ref = self.aux_ref(inputs_ref)
            x_illu = self.aux_illu(inputs_illu)
            x_all = torch.cat([x_depth,x_normal,x_ref,x_illu],dim=1)
            x_all = torch.sigmoid(x_all)
        else:
            print("dimension wrong!")
            exit()

        return x_all
