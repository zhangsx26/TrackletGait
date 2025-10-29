import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import  HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# from pytorch_wavelets import DWTForward
from ..torch_dwt.functional import dwt3,idwt3,dwt,dwt2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
                     
# https://github.com/KeKsBoTer/torch-dwt

class HWDownsampling_1D(nn.Module):
    def __init__(self, in_channel, out_channel, if_conv=True, if_dwt=True):
        super(HWDownsampling_1D, self).__init__()
        
        self.if_conv = if_conv
        self.if_dwt = if_dwt
        if self.if_conv:
            ch_in = in_channel * 2 if if_dwt else in_channel
            self.conv_bn_relu = nn.Sequential(
                nn.Conv3d(ch_in, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
            )
 
    def forward(self, x):
        batch, ch, frame, h, w = x.size() 
        if self.if_dwt:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
            x = dwt(x, 'haar')
            x = rearrange(x, '(b h w) n c f -> b (n c) f h w',b=batch,h=h,w=w)
            
        if self.if_conv:
            x = self.conv_bn_relu(x)
        return x
        
class HWDownsampling_2D(nn.Module):
    def __init__(self, in_channel, out_channel, if_conv=True):
        super(HWDownsampling_2D, self).__init__()
        
        self.if_conv = if_conv
        if self.if_conv:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )
 
    def forward(self, x):
        batch, ch, frame, h, w = x.size() 
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = dwt2(x, 'haar')
        
        if self.if_conv:
            x = rearrange(x, 'b n c h w -> b (n c) h w')
            x = self.conv_bn_relu(x)
            x = rearrange(x, '(b f) c h w -> b c f h w',b=batch,f=frame)
        else:
            x = rearrange(x, '(b f) n c h w -> b (n c) f h w',b=batch,f=frame)
        return x
               
class HWDownsampling_3D(nn.Module):
    def __init__(self, in_channel, out_channel, if_conv=True):
        super(HWDownsampling_3D, self).__init__()
        
        self.if_conv = if_conv
        if self.if_conv:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv3d(in_channel * 8, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
            )
 
    def forward(self, x):
        x = dwt3(x, 'haar')
        batch, n, ch, frame, h, w = x.size() 
        if self.if_conv:
            x = rearrange(x, 'b n c f h w -> b (n c) f h w')
            x = self.conv_bn_relu(x)
        else:
            x = rearrange(x, 'b n c f h w -> b (n c) f h w')
        return x
        
     
class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, h=32, if_tBranch=True):
        super(BasicBlock2D, self).__init__()

        self.if_tBranch = if_tBranch
        self.h = h
        self.relu = nn.ReLU(inplace=True)
        
        self.pooling = None
        if stride == [1,2,2]:
            self.conv1 = conv3x3(inplanes, inplanes)
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.pooling = HWDownsampling_2D(inplanes, planes)
        elif stride == [2,2,2]:
            self.conv1 = conv3x3(inplanes, inplanes)
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.pooling = HWDownsampling_3D(inplanes, planes)
        elif stride == [2,1,1]:
            self.conv1 = conv3x3(inplanes, inplanes)
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.pooling = HWDownsampling_1D(inplanes, planes)
        else:
            self.conv1 = conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            
            
        if self.if_tBranch:   
            # self.th = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=1) 
            self.th = nn.Conv3d(planes, planes, (7, 1, 1), (1, 1, 1), (3, 0, 0), bias=False, groups=4)
            self.thbn = nn.BatchNorm3d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
    
        batch, ch, frame, h, w = x.size() 
        identity = x

        x = rearrange(x, 'b c f h w -> (b f) c h w')

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.pooling is not None:
            out = rearrange(out, '(b f) c h w -> b c f h w',b=batch,f=frame)
            out = self.pooling(out) 
            batch, ch, frame, h, w = out.size() 
            out = rearrange(out, 'b c f h w -> (b f) c h w')
                
        if self.if_tBranch:   
            out = rearrange(out, '(b f) c h w -> b c f h w',b=batch,f=frame)
            batch, ch, frame, h, w = out.size() 
            out_th = self.th(out)
            out_th = self.thbn(out_th)

            out = out + out_th
            out = self.relu(out)

            out = rearrange(out, 'b c f h w -> (b f) c h w',b=batch)
            
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        identity = rearrange(identity, 'b c f h w -> (b f) c h w')

        out = out + identity
        out = self.relu(out)

        out = rearrange(out, '(b f) c h w -> b c f h w',b=batch)

        return out

     
def conv2x2(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride)       
        
        
        

class TrackletGait_s(BaseModel):   

    def build_network(self, model_cfg):

        
        self.pooling1 = HWDownsampling_1D(1, 16, if_dwt=False)
        self.inplanes = 16
        
        # self.layer1 = self.make_layer(BasicBlock2D, 64, [1,1,1], blocks_num=1, h=64, if_tBranch=False)
        
        self.layer2 = self.make_layer(BasicBlock2D, 256, [1,2,2], blocks_num=1, h=32, if_tBranch=True)
        # self.layer3 = self.make_layer(BasicBlock2D, 128, [1,1,1], blocks_num=1, h=32, if_tBranch=True)
        # self.layer4 = self.make_layer(BasicBlock2D, 128, [1,1,1], blocks_num=1, h=32, if_tBranch=True)
        # self.layer5 = self.make_layer(BasicBlock2D, 128, [1,1,1], blocks_num=1, h=32, if_tBranch=True)
        
        self.layer6 = self.make_layer(BasicBlock2D, 512, [1,2,2], blocks_num=1, h=16, if_tBranch=True)
        # self.layer7 = self.make_layer(BasicBlock2D, 256, [1,1,1], blocks_num=1, h=16, if_tBranch=True)
        # self.layer8 = self.make_layer(BasicBlock2D, 256, [1,1,1], blocks_num=1, h=16, if_tBranch=True)
        # self.layer9 = self.make_layer(BasicBlock2D, 256, [1,1,1], blocks_num=1, h=16, if_tBranch=True)
        
        # self.layer10 = self.make_layer(BasicBlock2D, 512, [1,1,1], blocks_num=1, h=16, if_tBranch=True)

        self.FCs = SeparateFCs(16, 512, 256)
        self.BNNecks = SeparateBNNecks(16, 256, class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

                        
    def make_layer(self, block, planes, stride, blocks_num, h=64, if_tBranch=False):

        #
        layers = []
        if stride == [1,2,2]:
            layers.append(  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))  )
        elif stride == [2,2,2]:
            layers.append(  nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))  )
        elif stride == [2,1,1]:
            layers.append(  nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))  )
        else:
            pass
        
        if self.inplanes != planes * block.expansion:
            layers.append(  nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0], bias=False)  )
            layers.append(  nn.BatchNorm3d(planes * block.expansion)  )
            
        downsample = nn.Sequential(*layers)

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, h=h, if_tBranch=if_tBranch)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)
        del ipts

        batch, ch, frame, h, w = sils.size() 
        if self.training:

            outs = self.pooling1(sils)

            # outs = self.layer1(outs)
            outs = self.layer2(outs)
            # outs = self.layer3(outs)
            # outs = self.layer4(outs)
            # outs = self.layer5(outs)  
            outs = self.layer6(outs)
            # outs = self.layer7(outs)
            # outs = self.layer8(outs)
            # outs = self.layer9(outs)
            # outs = self.layer10(outs)
              
            outs = reduce(outs,'b c f h w -> b c h w','max')        

            outs = self.HPP(outs)  # [n, c, p]
            embed_1 = self.FCs(outs)  # [n, c, p]
            _, logits = self.BNNecks(embed_1)  # [n, c, p]

            embed = embed_1

            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed_1, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')[:64],
                },
                'inference_feat': {
                }
            }
        
            return retval


            
        else:
        
            batch, ch, frame, h, w = sils.size() 
            outs = self.pooling1(sils)

            # outs = self.layer1(outs)
            outs = self.layer2(outs)
            # outs = self.layer3(outs)
            # outs = self.layer4(outs)
            # outs = self.layer5(outs)  
            outs = self.layer6(outs)
            # outs = self.layer7(outs)
            # outs = self.layer8(outs)
            # outs = self.layer9(outs)
            # outs = self.layer10(outs)

            outs = reduce(outs,'b c f h w -> b c h w','max')   
            feat = self.HPP(outs)  # [n, c, p]
            embed = self.FCs(feat)  # [n, c, p]

            retval = {
                'training_feat': {
                },
                'visual_summary': {
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
        
            return retval
        
