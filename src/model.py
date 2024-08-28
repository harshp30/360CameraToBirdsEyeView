import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
            # Second convolutional layer
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional block.
        """
        return self.conv(x)

class UNetXST(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        """
        Initialize the UNetXST model.

        Args:
            in_channels (int): Number of input channels (default: 12 for 4 views * 3 channels each).
            out_channels (int): Number of output channels (default: 3 for RGB).
        """
        super(UNetXST, self).__init__()

        # Encoder (downsampling) layers
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(2)

        # Middle (bottleneck) layer
        self.middle = ConvBlock(512, 1024)

        # Decoder (upsampling) layers
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # Transposed conv for upsampling
        self.decoder4 = ConvBlock(1024, 512)  # 1024 due to skip connection

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

        # Final convolutional layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """
        Forward pass of the UNetXST model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (predicted BEV image).
        """
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Middle (bottleneck)
        m = self.middle(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)  # Skip connection
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)  # Skip connection
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # Skip connection
        d1 = self.decoder1(d1)

        # Final convolution to produce output
        out = self.final_conv(d1)

        return out
