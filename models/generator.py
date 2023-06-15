import torch
import torch.nn as nn
import torch.functional as F
from collections import deque
from models.blocks import MappingNetwork, WSConv2d, StyleGanBlock, StyleGanInitBlock, ID_Layer
import logging

class StyleGan_Generator(nn.Module):
    def __init__(self, z_dim, w_dim, out_channels, latent_channels, leakyInReLU=0.2, start_size=(4, 4), scale_factor=2, alpha=0.9):
        super().__init__()
        self.z_dim, self.w_dim, self.out_channels, self.latent_channels, self.leakyInReLU, self.start_size, self.scale_factor, self.alpha = (
             z_dim,      w_dim,      out_channels,      latent_channels,      leakyInReLU,      start_size,      scale_factor,      alpha )
        assert len(latent_channels) >= 1, 'latent_channels must have at least 1 elements'
        
        self.map = ID_Layer(MappingNetwork(z_dim, w_dim))

        self.init_block = ID_Layer(StyleGanInitBlock(latent_channels[0], latent_channels[0], w_dim, start_size, leakyInReLU))
        
        self.feature_blocks, self.out_layers = (
            nn.ModuleList([]),
            nn.ModuleList([ID_Layer(WSConv2d(latent_channels[0], out_channels, kernel_size = 1, stride=1, padding=0))])
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        for in_c, out_c in zip(latent_channels[:-1], latent_channels[1:]):
            self.feature_blocks.append(ID_Layer(StyleGanBlock(in_c, out_c, w_dim)))
            self.out_layers.append(ID_Layer(WSConv2d(out_c, out_channels, kernel_size = 1, stride=1, padding=0)))
        # self.combine = ID_Layer(nn.Conv2d(2*out_channels, out_channels, kernel_size=1))
        self.layers = [self.map, self.init_block, *self.feature_blocks, *self.out_layers]
        
    def forward(self, noise):
        w = self.map(noise)
        x = self.init_block(w)
        y_hat = self.out_layers[0](x)

        for feature_block, out_layer in zip(self.feature_blocks, self.out_layers[1:]):
            x = self.upsample(x)
            x = feature_block(x, w)
            y_hat_up = self.upsample(y_hat)
            y_hat = out_layer(x)
            # y_hat = self.combine(torch.concat([y_hat_up, y_hat], dim=1))
            y_hat = self.alpha * y_hat + ( 1 - self.alpha ) * y_hat_up

        return y_hat
    
    def state_dict(self):
        state = {layer.id(): layer.state_dict() for layer in self.layers}
        return state
    
    def load_state_dict(self, state_dict):
        for layer in self.layers:
            if layer.id() in state_dict:
                layer.load_state_dict(state_dict[layer.id()])
            else:
                logging.debug(f'{self.__class__.__name__} {layer.id()[:80]}... not found in state_dict')
        
        return self
