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
class CONXA_Head_CITY(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, middle_channels=256, head_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, category_emb_dim = 32, **kwargs):
        super(CONXA_Head_CITY, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.middle_channels = middle_channels
        self.BatchNorm = norm_layer
        self.middle_channels = middle_channels
        self.head_channels = head_channels
        self.in_channel = category_emb_dim // 19

        self.fp16_enabled = False

        self.mlahead_road = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_road = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))         
       
        self.edge_road = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_sidewalk = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_sidewalk = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True)) 
      
        self.edge_sidewalk = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_building = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_building = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))    
       
        self.edge_building = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_wall = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_wall = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))   
        
        self.edge_wall = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_fence = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_fence = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))    
        
        self.edge_fence = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_pole = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_pole = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))  

        self.edge_pole = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_traffic_light = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_traffic_light = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))   
      
        self.edge_traffic_light = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_traffic_sign = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_traffic_sign = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_traffic_sign = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_vegetation = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_vegetation = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_vegetation = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_terrain = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_terrain = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_terrain = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_sky = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_sky = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_sky = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_person = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_person = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_person = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_rider = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_rider = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_rider = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_car = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_car = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_car = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_truck = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_truck = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_truck = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bus = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bus = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_bus = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_train = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_train = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_train = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_motorcycle = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_motorcycle = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_motorcycle = nn.Conv2d(self.head_channels, 1, 1)

        self.mlahead_bicycle = BIMLAHead_BN_ConvNeXt(in_channel=self.in_channel, middle_channels=self.middle_channels, head_channels=self.head_channels, norm_cfg=self.norm_cfg)
        self.global_features_bicycle = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1), build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True),
            nn.Conv2d(self.head_channels, self.head_channels, 1),build_norm_layer(self.norm_cfg, self.head_channels)[1], nn.ReLU(inplace=True))
        self.edge_bicycle = nn.Conv2d(self.head_channels, 1, 1)
        
    def forward(self, inputs):
    
        inputs = inputs.chunk(19, dim=1)
        
        road = inputs[0]
        sidewalk = inputs[1]
        building = inputs[2]
        wall = inputs[3]
        fence = inputs[4]
        pole = inputs[5]
        traffic_light = inputs[6]
        traffic_sign = inputs[7]
        vegetation = inputs[8]
        terrain = inputs[9]
        sky = inputs[10]
        person = inputs[11]
        rider = inputs[12]
        car = inputs[13]
        truck = inputs[14]
        bus = inputs[15]
        train = inputs[16]
        motorcycle = inputs[17]
        bicycle = inputs[18]
         
        x_road = self.mlahead_road(road)
        x_road = self.global_features_road(x_road)
        edge_road = self.edge_road(x_road)
        edge_road = torch.sigmoid(edge_road)
        
        x_sidewalk = self.mlahead_sidewalk(sidewalk)
        x_sidewalk = self.global_features_sidewalk(x_sidewalk)
        edge_sidewalk = self.edge_sidewalk(x_sidewalk)
        edge_sidewalk = torch.sigmoid(edge_sidewalk)
        
        x_building = self.mlahead_building(building)
        x_building = self.global_features_building(x_building)
        edge_building = self.edge_building(x_building)
        edge_building = torch.sigmoid(edge_building)
        
        x_wall = self.mlahead_wall(wall)
        x_wall = self.global_features_wall(x_wall)
        edge_wall = self.edge_wall(x_wall)
        edge_wall = torch.sigmoid(edge_wall)
        
        x_fence = self.mlahead_fence(fence)
        x_fence = self.global_features_fence(x_fence)
        edge_fence = self.edge_fence(x_fence)
        edge_fence = torch.sigmoid(edge_fence)
        
        x_pole = self.mlahead_pole(pole)
        x_pole = self.global_features_pole(x_pole)
        edge_pole = self.edge_pole(x_pole)
        edge_pole = torch.sigmoid(edge_pole)
        
        x_traffic_light = self.mlahead_traffic_light(traffic_light)
        x_traffic_light = self.global_features_traffic_light(x_traffic_light)
        edge_traffic_light = self.edge_traffic_light(x_traffic_light)
        edge_traffic_light = torch.sigmoid(edge_traffic_light)
        
        x_traffic_sign = self.mlahead_traffic_sign(traffic_sign)
        x_traffic_sign = self.global_features_traffic_sign(x_traffic_sign)
        edge_traffic_sign = self.edge_traffic_sign(x_traffic_sign)
        edge_traffic_sign = torch.sigmoid(edge_traffic_sign)
        
        x_vegetation = self.mlahead_vegetation(vegetation)
        x_vegetation = self.global_features_vegetation(x_vegetation)
        edge_vegetation = self.edge_vegetation(x_vegetation)
        edge_vegetation = torch.sigmoid(edge_vegetation)
        
        x_terrain = self.mlahead_terrain(terrain)
        x_terrain = self.global_features_terrain(x_terrain)
        edge_terrain = self.edge_terrain(x_terrain)
        edge_terrain = torch.sigmoid(edge_terrain)
        
        x_sky = self.mlahead_sky(sky)
        x_sky = self.global_features_sky(x_sky)
        edge_sky = self.edge_sky(x_sky)
        edge_sky = torch.sigmoid(edge_sky)
        
        x_person = self.mlahead_person(person)
        x_person = self.global_features_person(x_person)
        edge_person = self.edge_person(x_person)
        edge_person = torch.sigmoid(edge_person)
        
        x_rider = self.mlahead_rider(rider)
        x_rider = self.global_features_rider(x_rider)
        edge_rider = self.edge_rider(x_rider)
        edge_rider = torch.sigmoid(edge_rider)
        
        x_car = self.mlahead_car(car)
        x_car = self.global_features_car(x_car)
        edge_car = self.edge_car(x_car)
        edge_car = torch.sigmoid(edge_car)
        
        x_truck = self.mlahead_truck(truck)
        x_truck = self.global_features_truck(x_truck)
        edge_truck = self.edge_truck(x_truck)
        edge_truck = torch.sigmoid(edge_truck)
        
        x_bus = self.mlahead_bus(bus)
        x_bus = self.global_features_bus(x_bus)
        edge_bus = self.edge_bus(x_bus)
        edge_bus = torch.sigmoid(edge_bus)
        
        x_train = self.mlahead_train(train)
        x_train = self.global_features_train(x_train)
        edge_train = self.edge_train(x_train)
        edge_train = torch.sigmoid(edge_train)
        
        x_motorcycle = self.mlahead_motorcycle(motorcycle)
        x_motorcycle = self.global_features_motorcycle(x_motorcycle)
        edge_motorcycle = self.edge_motorcycle(x_motorcycle)
        edge_motorcycle = torch.sigmoid(edge_motorcycle)
        
        x_bicycle = self.mlahead_bicycle(bicycle)
        x_bicycle = self.global_features_bicycle(x_bicycle)
        edge_bicycle = self.edge_bicycle(x_bicycle)
        edge_bicycle = torch.sigmoid(edge_bicycle)
        
        x = torch.cat([
            x_road, x_sidewalk, x_building, x_wall,
            x_fence, x_pole, x_traffic_light, x_traffic_sign, x_vegetation, x_terrain, x_sky,
            x_person, x_rider, x_car, x_truck, x_bus,
            x_train, x_motorcycle, x_bicycle
        ], dim=1)
        
        edge = torch.cat([
            edge_road, edge_sidewalk, edge_building, edge_wall,
            edge_fence, edge_pole, edge_traffic_light, edge_traffic_sign, edge_vegetation, edge_terrain, edge_sky,
            edge_person, edge_rider, edge_car, edge_truck, edge_bus,
            edge_train, edge_motorcycle, edge_bicycle
        ], dim=1)
        
        return edge, x

