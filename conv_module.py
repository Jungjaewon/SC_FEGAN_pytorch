import torch
import torch.nn as nn
import torch.nn.functional as F

from gated_module import GatedConv2d
from gated_module import TransposeGatedConv2d
# Reference : https://github.com/milesial/Pytorch-UNet


class ConvMod(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, LR=0.1, single=False, first=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if single:
            if first:
                self.conv = nn.Sequential(
                    GatedConv2d(in_channels, mid_channels, kernel_size=7, padding=3),
                    #nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(LR, inplace=True),
                )
            else:
                self.conv = nn.Sequential(
                    GatedConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(LR, inplace=True),
                )
        else:
            self.conv = nn.Sequential(
                GatedConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(LR, inplace=True),
                GatedConv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(LR, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, LR=0.1, single=False):
        super().__init__()
        self.main = nn.Sequential(
            GatedConv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            ConvMod(in_channels, out_channels, LR=LR, single=single)
        )

    def forward(self, x):
        return self.main(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, channel1, channel2, bilinear=False, LR=0.1, single=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            raise NotImplemented
            #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            #self.up = TransposeGatedConv2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = TransposeGatedConv2d(channel1, channel1, kernel_size=3, stride=1, padding=1)
            self.conv = ConvMod(channel1 + channel2, channel2, LR=LR, single=single)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv = GatedConv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.conv(x))