from torch import nn
from torch.utils.data import Dataset

# Note [-1 : 1] -> nn.Tanh
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 256 -> 128 -> 64 -> 32 -> 16 -> 7 -> 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # N, 128
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # N, 64
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # N, 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # N, 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=16), # N, 1
            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=16), # N, 1
            nn.ReLU(), 
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),# N, 16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
            
        
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Note [-1 : 1] -> nn.Tanh
class Autoencoder_v2(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 256 -> 128 -> 64 -> 32 -> 16 -> 7 -> 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # N, 128
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # N, 64
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # N, 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # N, 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # N, 8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=8), # N, 1
            nn.ReLU()

 
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=8), # N, 1
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 1
            nn.ReLU(), 
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),# N, 16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16
#             nn.AvgPool2d(kernel_size=49, stride=1, padding=24)
            
        
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded    

  
    
# Note [-1 : 1] -> nn.Tanh
# nn.MaxPool2d -> nn.MaxUnPool2d


class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: (256, 16, 16)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Output: (512, 8, 8)
            nn.ReLU(),
            nn.Flatten(), # Flatten the output
            nn.Linear(512 * 8 * 8, 1024), # Latent vector
            nn.ReLU()
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512 * 8 * 8), # Latent vector to feature map
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)), # Reshape to 4D tensor
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (256, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (1, 256, 256)
#             nn.Sigmoid()  # Apply Sigmoid activation to ensure output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
import torch
import torch.nn as nn

class VGG_simple(nn.Module):
    def __init__(self):
        super(VGG_simple, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256 -> 128x128
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Dropout(0.25),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256 * 32 * 32),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 32, 32)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  # output image [0, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256 -> 128x128
            nn.Dropout(0.25),
            
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64
            nn.Dropout(0.25),

            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Dropout(0.25),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256 * 32 * 32),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 32, 32)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # output image [0, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# 모델 생성

    
# Autoencoder 정의
class VGG_deep(nn.Module):
    def __init__(self):
        super(VGG_deep, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256 -> 128x128
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(512 * 32 * 32, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024)  # Bottleneck
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512 * 32 * 32),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 32, 32)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  # output image [0, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
