
import torch
from torch import nn


class Parameter(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = nn.Parameter(parameter)
    def forward(self, *args):
        return self.parameter
class ID_Layer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.pretrained = False
        
    def id(self):
        return super().__str__().replace('\n', '')
    
    def forward(self, *args, **kwargs):
        return self.module( *args, **kwargs)
    

class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale  = (2/in_features) ** 0.5
        self.bias   = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.linear(x * self.scale) + self.bias
    
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    def forward(self,x ):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+  self.epsilon)
    
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )
    
    def forward(self,x):
        return self.mapping(x)
    
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale   = WSLinear(w_dim, channels)
        self.style_bias    = WSLinear(w_dim, channels)

    def forward(self,x,w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias  = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias
    

class InjectNoise(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x):
        b, c, w, h = x.shape
        noise = torch.randn((b, 1, w, h), device = x.device, dtype=x.dtype)
        return x + self.weight * noise

class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class StyleGanInitBlock(nn.Module):
    def __init__(self, in_channel, out_channel, w_dim, leakyInReLU=0.2):
        super().__init__()
        self.in_channel, self.out_channel, self.w_dim, self.leakyInReLU = in_channel, out_channel, w_dim, leakyInReLU
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(leakyInReLU, inplace=True)
        self.inject_noise1 = InjectNoise(out_channel)
        self.inject_noise2 = InjectNoise(out_channel)
        self.adain1 = AdaIN(out_channel, w_dim)
        self.adain2 = AdaIN(out_channel, w_dim)

    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(x)), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv1(x))), w)
        return x
    
class StyleGanBlock(nn.Module):
    def __init__(self, in_channel, out_channel, w_dim, leakyInReLU=0.2):
        super().__init__()
        self.in_channel, self.out_channel, self.w_dim, self.leakyInReLU = in_channel, out_channel, w_dim, leakyInReLU
        self.conv1 = WSConv2d(in_channel, out_channel)
        self.conv2 = WSConv2d(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(leakyInReLU, inplace=True)
        self.inject_noise1 = InjectNoise(out_channel)
        self.inject_noise2 = InjectNoise(out_channel)
        self.adain1 = AdaIN(out_channel, w_dim)
        self.adain2 = AdaIN(out_channel, w_dim)

    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x
    
class MiniBatchSTD(nn.Module):
    """Minibatch standard deviation layer for the discriminator"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b, c, h, w = x.shape
        std = (
            torch.std(x, dim=0)
                .mean()
                .repeat(b, 1, h, w)
        )
        return torch.cat([x, std], dim=1)
    
class MiniBatchStdLayer(nn.Module):
    """Minibatch standard deviation layer for the discriminator"""
    def __init__(self, group_size=4, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x