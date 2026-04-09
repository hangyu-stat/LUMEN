""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False, attention_module=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.bilinear = bilinear
        '''
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, attention_module=attention_module))
        self.down2 = (Down(128, 256, attention_module=attention_module))
        self.down3 = (Down(256, 512, attention_module=attention_module))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, attention_module=attention_module))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        '''

        '''
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, attention_module=attention_module))
        self.down2 = (Down(128, 256, attention_module=attention_module))
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, 512 // factor, attention_module=attention_module))

        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        '''
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, attention_module=attention_module))
        factor = 2 if bilinear else 1
        self.down2 = (Down(128, 256 // factor, attention_module=attention_module))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64, bilinear))

        #self.outc = (OutConv(64, n_classes))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    unet = UNet(1, 10)
    image = torch.zeros(2, 1, 256, 256)
    device = torch.device("cuda:{}".format(0))
    unet.to(device)
    image = image.to(device)
    unet(image)