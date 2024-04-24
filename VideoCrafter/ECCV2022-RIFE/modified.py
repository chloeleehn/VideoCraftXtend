import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, num_layers = 8):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        # self.convblock = nn.Sequential(
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        #     conv(c, c),
        # )
        # modified
        self.convblock = nn.Sequential(
            *[conv(c, c, dilation=2 ** i) for i in range(num_layers)]
        )
        # modified
        self.intermediate_conv = nn.Sequential(
            conv(c, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        # x = self.conv0(x)
        # x = self.convblock(x) + x
        # modified
        x = self.conv0(x)
        x_res = self.convblock(x)
        x = x_res + x
        x = self.intermediate_conv(x)
        x = x_res + x

        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask