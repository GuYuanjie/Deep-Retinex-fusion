import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out




class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(3//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.pad2 = nn.ReflectionPad2d(3 // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU()
        self.pad3 = nn.ReflectionPad2d(3 // 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu2 = nn.ELU()


    def forward(self, x):
        x=self.pad1(x)
        x=self.conv1(x)
        return self.elu2(self.bn2(self.conv3(self.pad3(self.elu1(self.bn1(self.conv2(self.pad2(x))))))))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.pad1 = nn.ReflectionPad2d(3//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.pad2 = nn.ReflectionPad2d(3 // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU()
        self.pad3 = nn.ReflectionPad2d(3 // 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu2 = nn.ELU()


    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return self.elu2(self.bn2(self.conv3(self.pad3(self.elu1(self.bn1(self.conv2(self.pad2(self.conv1(self.pad1(x))))))))))


class ZipperNet(nn.Module):
    def __init__(self):
        super(ZipperNet, self).__init__()
        depth = 1
        self.encoderblockx1 = EncoderBlock(depth, 8)
        self.encoderblocky1 = EncoderBlock(depth, 8)
        self.encoderblock2 = EncoderBlock(8, 16)
        self.encoderblock3 = EncoderBlock(16, 32)
        self.encoderblock4 = EncoderBlock(32, 64)
        self.encoderblock5 = EncoderBlock(64, 128)


        self.decoderblock1 = DecoderBlock(128, 64)
        self.decoderblock2 = DecoderBlock(64*2, 32)
        self.decoderblock3 = DecoderBlock(32*2, 16)
        self.decoderblock4 = DecoderBlock(16*2, 8)
        self.decoderblock5 = DecoderBlock(8*2, 8)

        self.CoordAtt = CoordAtt(256,256)

        self.conv1x1_1 = nn.Conv2d(16, 8, 1)
        self.conv1x1_2 = nn.Conv2d(32, 16, 1)
        self.conv1x1_3 = nn.Conv2d(64, 32, 1)
        self.conv1x1_4 = nn.Conv2d(128, 64, 1)
        self.conv1x1_5 = nn.Conv2d(256, 128, 1)

        self.conv3x3 = nn.Conv2d(8,depth,3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,y):

        layerx1 = self.encoderblockx1(x)
        layery1 = self.encoderblocky1(y)

        cat1 = self.conv1x1_1(torch.cat([layerx1,layery1],1))

        layerx2 = self.encoderblock2(layerx1+layery1)
        layery2 = self.encoderblock2(layery1)

        cat2 = self.conv1x1_2(torch.cat([layerx2, layery2], 1))

        layerx3 = self.encoderblock3(layerx2)
        layery3 = self.encoderblock3(layery2+layerx2)

        cat3 = self.conv1x1_3(torch.cat([layerx3, layery3], 1))

        layerx4 = self.encoderblock4(layerx3+layery3)
        layery4 = self.encoderblock4(layery3)

        cat4 = self.conv1x1_4(torch.cat([layerx4, layery4], 1))

        layerx5 = self.encoderblock5(layerx4)
        layery5 = self.encoderblock5(layery4+layerx4)

        up1 = self.decoderblock1(self.conv1x1_5(torch.cat([layerx5, layery5], 1)))

        up2 = self.decoderblock2(torch.cat([up1, cat4], 1))
        up3 = self.decoderblock3(torch.cat([up2, cat3], 1))
        up4 = self.decoderblock4(torch.cat([up3, cat2], 1))
        output = self.decoderblock5(torch.cat([up4, cat1], 1))
        output = self.conv3x3(output)
        output = self.sigmoid(output)
        output = F.interpolate(output, (x.shape[2],x.shape[3]), mode='bicubic', align_corners=False)
        return output
