import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data

def conv(in_planes, output_channels, kernel_size, stride, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                           stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )

def output_layer(input_channels, output_channels, kernel_size, stride, dropout_rate):
    return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2)

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels

        # Encoder
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

        # Decoder
        self.deconv2 = deconv(256, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
        self.output_layer = output_layer(16 + input_channels, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        # x is expected to be of shape (B, T, C, H, W)
        B, T, C, H, W = x.size()
        assert T == 1, "Expected T=1 since TCHW=1 2 64 448"
        x = x.squeeze(1)  # Remove the T dimension; x is now of shape (B, C, H, W)

        # Encoder path (downsampling)
        out_conv1 = self.conv1(x)  # (B, 64, H/2, W/2)
        out_conv2 = self.conv2(out_conv1)  # (B, 128, H/4, W/4)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # (B, 256, H/8, W/8)

        # Decoder path (upsampling)
        out_deconv2 = self.deconv2(out_conv3)  # (B, 64, H/4, W/4)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)  # (B, 192, H/4, W/4)
        out_deconv1 = self.deconv1(concat2)  # (B, 32, H/2, W/2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)  # (B, 96, H/2, W/2)
        out_deconv0 = self.deconv0(concat1)  # (B, 16, H, W)
        concat0 = torch.cat((x, out_deconv0), 1)  # (B, 16 + input_channels, H, W)
        out = self.output_layer(concat0)  # (B, output_channels, H, W)

        out = out.unsqueeze(1)  # Add the T dimension back; now out is of shape (B, T, output_channels, H, W)
        return out

# input_channels = 1     # As per your input dimension C=2
# output_channels = 1    # Assuming you want the output to have the same number of channels
# kernel_size = 3        # You can adjust this value as needed
# dropout_rate = 0.5     # You can adjust this value as needed

# model = U_net(input_channels, output_channels, kernel_size, dropout_rate)

# # Example input tensor with dimensions BTCHW = (Batch, Time=1, Channels=2, Height=64, Width=448)
# B = 4   # Example batch size
# T = 1
# C = 1
# H = 128
# W = 128
# input_tensor = torch.randn(B, T, C, H, W)

# # Pass the input tensor through the model
# output = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")   # Should print torch.Size([B, 1, 2, 64, 448])
# print(f"Output shape: {output.shape}")        # Should print torch.Size([B, 1, 2, 64, 448])