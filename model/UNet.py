import torch
from torch import nn
import numpy as np
class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        dilations = [1, 2, 4, 8]  # use
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=dilations[0], dilation=dilations[0])
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=dilations[1], dilation=dilations[1])
        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=dilations[2], dilation=dilations[2])
        self.conv4 = nn.Conv2d(in_c, out_c, 3, padding=dilations[3], dilation=dilations[3])


        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_c, out_c, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_c * 5, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv_pool(self.pool(x)).expand_as(x1)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.fuse(out)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 3 classes (LV, RV, Myocardium)

        self.global_branch = nn.Sequential(
            ASPP(64, 64),  # ASPP 增强全局感受野ASPP(16, 16)

        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_aspp  = self.global_branch(enc1)

        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(kernel_size=2)(enc3))

        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1_aspp), dim=1))

        return self.out_conv(dec1)


# from PIL import Image
# aaa = Image.fromarray(np.array(enhanced_filter[0,0].cpu().detach().numpy()*255 , dtype=np.uint8))
# aaa.show()
# aaa.save("11.jpg")