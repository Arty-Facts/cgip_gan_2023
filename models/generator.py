import torch
import torch.nn as nn
import torch.functional as F
from collections import deque
from models.blocks import MappingNetwork, ConvBlock, StyleGanBlock, StyleGanInitBlock, ID_Layer
import logging

class StyleGan_Generator(nn.Module):
    def __init__(self, z_dim, w_dim, latent_channels, leakyInReLU=0.2, start_size=(4, 4)):
        super().__init__()
        self.z_dim, self.w_dim, self.latent_channels, self.factors, self.leakyInReLU, self.start_size = z_dim, w_dim, latent_channels, leakyInReLU, start_size
        assert len(latent_channels) >= 2, 'latent_channels must have at least 2 elements'
        
        h, w = start_size
        self.starting_cte = ID_Layer(nn.Parameter(torch.ones(1, latent_channels[0], h, w)))
        self.map = ID_Layer(MappingNetwork(z_dim, w_dim))

        self.feature_blocks, self.out_layers = (
            nn.ModuleList([ID_Layer(StyleGanInitBlock(latent_channels[0], latent_channels[1], w_dim, leakyInReLU))]),
            nn.ModuleList([ID_Layer(ConvBlock(latent_channels[0], latent_channels[1], kernel_size = 1, stride=1, padding=0))])
        )

        for in_c, out_c in zip(latent_channels[1:-1], latent_channels[2:]):
            self.feature_blocks.append(ID_Layer(nn.UpsamplingBilinear2d(scale_factor=2), StyleGanBlock(in_c, out_c, w_dim)))
            self.out_layers.append(ID_Layer(ConvBlock(out_c, latent_channels[-1], kernel_size = 1, stride=1, padding=0)))
        self.layers = [self.map, self.starting_cte, *self.feature_blocks, *self.out_layers]
        
    def forward(self, noise):
        w = self.map(noise)
        x = self.starting_cte

        for gan_block in self.feature_blocks:
            x = gan_block(x, w)

        y_hat = self.out_layers[-1](x)
        return y_hat
    
    def state_dict(self):
        state = {layer.id(): layer.state_dict() for layer in self.layers}
        return state
    
    def load_state_dict(self, state_dict):
        for layer in self.layers:
            if layer.id() in state_dict:
                layer.load_state_dict(state_dict[layer.id()])
            else:
                logging.warning(f'{layer.id()[:80]}... not found in state_dict')
        
        return self
