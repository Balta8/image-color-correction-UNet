import torch
from torch import nn
from UNet_Block import UNetBlock

class UNet(nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder 
        self.enc1 = UNetBlock(3, 64, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = UNetBlock(256, 512, dropout=0.3)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck 
        self.bottleneck = UNetBlock(512, 1024, dropout=0.3)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Sigmoid()  # [0,1]

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder + skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return self.activation(out)