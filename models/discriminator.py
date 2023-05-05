import torch
import torch.nn as nn
from models.blocks import WSConvLeakyBlock, WSConvReluBlock, MiniBatchSTD, ID_Layer
import logging

class Discriminator(nn.Module):
    def __init__(self, 
                 in_channels, 
                 latent_channels, 
                 Conv,
                 Pooling,
                 activation,
                 ):
        super().__init__()
        self.in_channels, self.latent_channels = in_channels, latent_channels
        assert len(latent_channels) >= 1, "latent_channels must have at least 1 elements"
        self.in_blocks = nn.ModuleList(
            [
                ID_Layer(
                    Conv(in_channels, out_c)
                    ) 
                for out_c in latent_channels
            ])
        self.feature_blocks = nn.ModuleList(
            [
                ID_Layer(
                    nn.Sequential(
                        Pooling(kernel_size=2, stride=2), 
                        Conv(in_c, out_c)
                        )
                    ) 
                    for in_c, out_c in zip(latent_channels[:-1], latent_channels[1:])
            ])
        self.activation = activation
        # this is the block for 4x4 input size
        self.final_block = ID_Layer(nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            MiniBatchSTD(),
            Conv(latent_channels[-1] + 1, latent_channels[-1], kernel_size=3, padding=1),
            self.activation,
            Conv(latent_channels[-1], latent_channels[-1], kernel_size=4, padding=0, stride=1),
            self.activation,
            Conv(
                latent_channels[-1], 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
            nn.Flatten(start_dim=1)
        ))
        self.layers = [*self.in_blocks, *self.feature_blocks, self.final_block]

    def embed(self, x):
        x = self.in_blocks[0](x)
        for block in self.feature_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        x = self.embed(x)
        return self.final_block(x)
    
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


def discriminator_maxpool(in_channels, latent_channels, leakyInReLU=0.2):
    return Discriminator(in_channels, latent_channels, 
                         WSConvLeakyBlock, nn.MaxPool2d, 
                         nn.LeakyReLU(leakyInReLU))

def discriminator_avgpool(in_channels, latent_channels, leakyInReLU=0.2):
    return Discriminator(in_channels, latent_channels, 
                         WSConvLeakyBlock, nn.AvgPool2d, 
                         nn.LeakyReLU(leakyInReLU))

def discriminator_stride_conv(in_channels, latent_channels, leakyInReLU=0.2):
    return Discriminator(in_channels, latent_channels, 
                         WSConvLeakyBlock, nn.Conv2d, 
                         nn.LeakyReLU(leakyInReLU))