import torch

class MeanDiscriminatorLoss():
    """Mean of the discriminator output for face images"""
    def __call__(self, fake, discriminator, device):
        return -torch.mean(discriminator(fake))
