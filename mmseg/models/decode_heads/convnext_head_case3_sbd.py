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
class ConvNeXt_Head_CASE3_SBD(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, middle_channels=256, head_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, category_emb_dim = 32, **kwargs):
        super(ConvNeXt_Head_CASE3_SBD, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.middle_channels = middle_channels
        self.BatchNorm = norm_layer
        self.middle_channels = middle_channels
        self.head_channels = head_channels
        self.in_channel = category_emb_dim // 20 

        self.fp16_enabled = False

        self.mlahead_aeroplane = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_aeroplane = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
                     
        
        self.edge_aeroplane = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bicycle = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bicycle = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        
        self.edge_bicycle = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bird = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bird = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        
        self.edge_bird = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_boat = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_boat = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        
        self.edge_boat = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bottle = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bottle = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        
        self.edge_bottle = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bus = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bus = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
            
        self.edge_bus = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_car = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_car = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_car = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_cat = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_cat = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_cat = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_chair = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_chair = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_chair = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_cow = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_cow = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_cow = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_table = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_table = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_table = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_dog = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_dog = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_dog = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_horse = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_horse = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_horse = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_motorbike = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_motorbike = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_motorbike = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_person = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_person = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_person = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_pottedplant = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_pottedplant = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_pottedplant = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_sheep = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_sheep = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_sheep = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_sofa = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_sofa = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_sofa = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_train = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_train = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_train = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_tvmonitor = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_tvmonitor = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_tvmonitor = nn.Conv2d(self.head_channels, 1, 1)


        
        
    def forward(self, inputs):
    
        inputs = inputs.chunk(20, dim = 1)

        
        aeroplane = inputs[0]
        bicycle = inputs[1]
        bird = inputs[2]
        boat = inputs[3]
        bottle = inputs[4]
        bus = inputs[5]
        car = inputs[6]
        cat = inputs[7]
        chair = inputs[8]
        cow = inputs[9]
        table = inputs[10]
        dog = inputs[11]
        horse = inputs[12]
        motorbike = inputs[13]
        person = inputs[14]
        pottedplant = inputs[15]
        sheep = inputs[16]
        sofa = inputs[17]
        train = inputs[18]
        tvmonitor = inputs[19]
         
        

        x_aeroplane = self.mlahead_aeroplane(aeroplane)
        x_aeroplane = self.global_features_aeroplane(x_aeroplane)
        edge_aeroplane = self.edge_aeroplane(x_aeroplane)
        edge_aeroplane = torch.sigmoid(edge_aeroplane)
        
        x_bicycle = self.mlahead_bicycle(bicycle)
        x_bicycle = self.global_features_bicycle(x_bicycle)
        edge_bicycle = self.edge_bicycle(x_bicycle)
        edge_bicycle = torch.sigmoid(edge_bicycle)
        
        x_bird = self.mlahead_bird(bird)
        x_bird = self.global_features_bird(x_bird)
        edge_bird = self.edge_bird(x_bird)
        edge_bird = torch.sigmoid(edge_bird)
        
        x_boat = self.mlahead_boat(boat)
        x_boat = self.global_features_boat(x_boat)
        edge_boat = self.edge_boat(x_boat)
        edge_boat = torch.sigmoid(edge_boat)
        
        x_bottle = self.mlahead_bottle(bottle)
        x_bottle = self.global_features_bottle(x_bottle)
        edge_bottle = self.edge_bottle(x_bottle)
        edge_bottle = torch.sigmoid(edge_bottle)
        
        x_bus = self.mlahead_bus(bus)
        x_bus = self.global_features_bus(x_bus)
        edge_bus = self.edge_bus(x_bus)
        edge_bus = torch.sigmoid(edge_bus)
        
        x_car = self.mlahead_car(car)
        x_car = self.global_features_car(x_car)
        edge_car = self.edge_car(x_car)
        edge_car = torch.sigmoid(edge_car)
        
        x_cat = self.mlahead_cat(cat)
        x_cat = self.global_features_cat(x_cat)
        edge_cat = self.edge_cat(x_cat)
        edge_cat = torch.sigmoid(edge_cat)
        
        x_chair = self.mlahead_chair(chair)
        x_chair = self.global_features_chair(x_chair)
        edge_chair = self.edge_chair(x_chair)
        edge_chair = torch.sigmoid(edge_chair)
        
        x_cow = self.mlahead_cow(cow)
        x_cow = self.global_features_cow(x_cow)
        edge_cow = self.edge_cow(x_cow)
        edge_cow = torch.sigmoid(edge_cow)
        
        x_table = self.mlahead_table(table)
        x_table = self.global_features_table(x_table)
        edge_table = self.edge_table(x_table)
        edge_table = torch.sigmoid(edge_table)
        
        x_dog = self.mlahead_dog(dog)
        x_dog = self.global_features_dog(x_dog)
        edge_dog = self.edge_dog(x_dog)
        edge_dog = torch.sigmoid(edge_dog)
        
        x_horse = self.mlahead_horse(horse)
        x_horse = self.global_features_horse(x_horse)
        edge_horse = self.edge_horse(x_horse)
        edge_horse = torch.sigmoid(edge_horse)
        
        x_motorbike = self.mlahead_motorbike(motorbike)
        x_motorbike = self.global_features_motorbike(x_motorbike)
        edge_motorbike = self.edge_motorbike(x_motorbike)
        edge_motorbike = torch.sigmoid(edge_motorbike)
        
        x_person = self.mlahead_person(person)
        x_person = self.global_features_person(x_person)
        edge_person = self.edge_person(x_person)
        edge_person = torch.sigmoid(edge_person)
        
        x_pottedplant = self.mlahead_pottedplant(pottedplant)
        x_pottedplant = self.global_features_pottedplant(x_pottedplant)
        edge_pottedplant = self.edge_pottedplant(x_pottedplant)
        edge_pottedplant = torch.sigmoid(edge_pottedplant)
        
        x_sheep = self.mlahead_sheep(sheep)
        x_sheep = self.global_features_sheep(x_sheep)
        edge_sheep = self.edge_sheep(x_sheep)
        edge_sheep = torch.sigmoid(edge_sheep)
        
        x_sofa = self.mlahead_sofa(sofa)
        x_sofa = self.global_features_sofa(x_sofa)
        edge_sofa = self.edge_sofa(x_sofa)
        edge_sofa = torch.sigmoid(edge_sofa)
        
        x_train = self.mlahead_train(train)
        x_train = self.global_features_train(x_train)
        edge_train = self.edge_train(x_train)
        edge_train = torch.sigmoid(edge_train)
        
        
        x_tvmonitor = self.mlahead_tvmonitor(tvmonitor)
        x_tvmonitor = self.global_features_tvmonitor(x_tvmonitor)
        edge_tvmonitor = self.edge_tvmonitor(x_tvmonitor)
        edge_tvmonitor = torch.sigmoid(edge_tvmonitor)
        
              

        x = torch.cat([
            x_aeroplane, x_bicycle, x_bird, x_boat,
            x_bottle, x_bus, x_car, x_cat, x_chair, x_cow, x_table,
            x_dog, x_horse, x_motorbike, x_person, x_pottedplant,
            x_sheep, x_sofa, x_train, x_tvmonitor
        ], dim=1)
        
        edge = torch.cat([
            edge_aeroplane, edge_bicycle, edge_bird, edge_boat,
            edge_bottle, edge_bus, edge_car, edge_cat, edge_chair, edge_cow, edge_table,
            edge_dog, edge_horse, edge_motorbike, edge_person, edge_pottedplant,
            edge_sheep, edge_sofa, edge_train, edge_tvmonitor
        ], dim=1)
        
        
        return edge, x
