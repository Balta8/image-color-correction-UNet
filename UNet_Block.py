import torch.nn as nn

# Define the U-Net block

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(UNetBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))  # dropout layer for regularization
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)