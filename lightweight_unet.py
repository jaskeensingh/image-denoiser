import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class LightweightUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LightweightUNet, self).__init__()
        self.encoder1 = DepthwiseSeparableConv(input_channels, 64)
        self.encoder2 = DepthwiseSeparableConv(64, 128)
        self.encoder3 = DepthwiseSeparableConv(128, 256)

        self.middle = DepthwiseSeparableConv(256, 512)

        self.decoder3 = DepthwiseSeparableConv(512, 256)
        self.decoder2 = DepthwiseSeparableConv(256, 128)
        self.decoder1 = DepthwiseSeparableConv(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        m = self.middle(e3)
        
        d3 = self.decoder3(m + e3)  # Skip connection
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        
        return self.final(d1)

if __name__ == "__main__":
    model = LightweightUNet(3, 1)  # Example for 3 input channels and 1 output class
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)
