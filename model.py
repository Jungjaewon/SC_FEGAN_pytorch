import torch.nn as nn
import torch

from conv_module import *
from gated_module import GatedConv2d
from spectral_normalization import SpectralNorm


class Generator(nn.Module):
    def __init__(self, in_ch=9, out_ch=3, bilinear=False, LR=0.1, single_conv=False):
        super(Generator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bilinear = bilinear
        self.dilated_gconv = list()

        # 512 256 128 64 32 16 8 4
        for i in range(4): # 0 1 2 3
            self.dilated_gconv.append(GatedConv2d(512, 512, kernel_size=3, padding=i + 1, dilation=i + 1))
            self.dilated_gconv.append(nn.InstanceNorm2d(512))
            self.dilated_gconv.append(nn.LeakyReLU(LR, inplace=True))

        self.inc = ConvMod(in_ch, 64, LR=LR, single=single_conv, first=True)
        self.down1 = Down(64, 128, LR=LR, single=single_conv) # 256
        self.down2 = Down(128, 256, LR=LR, single=single_conv) # 128
        self.down3 = Down(256, 512, LR=LR, single=single_conv) # 64
        self.down4 = Down(512, 512, LR=LR, single=single_conv) # 32
        self.down5 = Down(512, 512, LR=LR, single=single_conv) # 16
        self.down6 = Down(512, 512, LR=LR, single=single_conv) # 8
        self.down7 = Down(512, 512, LR=LR, single=single_conv) # 4
        self.up1 = Up(512, 512, LR=LR, single=single_conv)
        self.up2 = Up(512, 512, LR=LR, single=single_conv)
        self.up3 = Up(512, 512, LR=LR, single=single_conv)
        self.up4 = Up(512, 512, LR=LR, single=single_conv)
        self.up5 = Up(512, 256, LR=LR, single=single_conv)
        self.up6 = Up(256, 128, LR=LR, single=single_conv)
        self.up7 = Up(128, 64, LR=LR, single=single_conv)
        self.outc = OutConv(64 + in_ch, out_ch)
        self.dilated_gconv = nn.Sequential(*self.dilated_gconv)

    def forward(self, data):
        x1 = self.inc(data)
        x2 = self.down1(x1) # 256
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 64
        x5 = self.down4(x4) # 32
        x6 = self.down5(x5) # 16
        x7 = self.down6(x6) # 8
        x8 = self.down7(x7) # 4
        x8 = self.dilated_gconv(x8) # 4
        x = self.up1(x8, x7) # 8
        x = self.up2(x, x6) # 16
        x = self.up3(x, x5) # 32
        x = self.up4(x, x4) # 64
        x = self.up5(x, x3) # x3 is 256dim, x(up4) is 512dim # 128
        x = self.up6(x, x2) # 256
        x = self.up7(x, x1) # 512
        return self.outc(torch.cat([x, data], dim=1))


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=512, start_dim=9, conv_dim=64, repeat_n=5, LR=0.1):
        super(Discriminator, self).__init__()
        layers = list()
        layers.append(SpectralNorm(nn.Conv2d(start_dim, conv_dim, kernel_size=7, stride=1, padding=3)))
        #layers.append(nn.InstanceNorm2d(conv_dim)) # BatchNorm2d InstanceNorm2d
        layers.append(nn.LeakyReLU(LR, inplace=True))

        layers.append(SpectralNorm(nn.Conv2d(conv_dim, 128, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        layers.append(SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.InstanceNorm2d(256, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        layers.append(SpectralNorm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        layers.append(SpectralNorm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        layers.append(SpectralNorm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)))
        #layers.append(nn.InstanceNorm2d(256))
        #layers.append(nn.LeakyReLU(LR, inplace=True))

        """
        curr_dim = conv_dim
        for i in range(repeat_n):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = curr_dim * 2
        """

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
