import torch
import torch.nn as nn
from models.blocks import WSConv2d, ConvBlock, MiniBatchSTD, ID_Layer
import logging

class StyleGan_Discriminator(nn.Module):
    def __init__(self, in_channels, latent_channels, leakyInReLU=0.2):
        super().__init__()
        self.in_channels, self.latent_channels, self.leakyInReLU = in_channels, latent_channels, leakyInReLU
        assert len(latent_channels) >= 1, "latent_channels must have at least 1 elements"
        self.in_blocks = nn.ModuleList([ID_Layer(ConvBlock(in_channels, out_c)) for out_c in latent_channels])
        self.feature_blocks = nn.ModuleList([ID_Layer(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(in_c, out_c))) for in_c, out_c in zip(latent_channels[:-1], latent_channels[1:])])

        # this is the block for 4x4 input size
        self.final_block = ID_Layer(nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            MiniBatchSTD(),
            WSConv2d(latent_channels[-1] + 1, latent_channels[-1], kernel_size=3, padding=1),
            nn.LeakyReLU(leakyInReLU),
            WSConv2d(latent_channels[-1], latent_channels[-1], kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(leakyInReLU),
            WSConv2d(
                latent_channels[-1], 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
            nn.Flatten(start_dim=1)
        ))
        self.layers = [*self.in_blocks, *self.feature_blocks, self.final_block]

    def forward(self, x):
        x = self.in_blocks[0](x)
        for block in self.feature_blocks:
            x = block(x)
        return self.final_block(x)
    
    def state_dict(self):
        state = {layer.id(): layer.state_dict() for layer in self.layers}
        return state
        
    def load_state_dict(self, state_dict):
        for layer in self.layers:
            if layer.id() in state_dict:
                layer.load_state_dict(state_dict[layer.id()])
                logging.warning(f'{self.__class__.__name__} {layer.id()[:80]}... was found in state_dict')
            else:
                logging.warning(f'{self.__class__.__name__} {layer.id()[:80]}... not found in state_dict')
        return self
