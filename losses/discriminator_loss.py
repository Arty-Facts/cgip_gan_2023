import torch


class GradientPenalty():
    def __init__(self, lambda_gb=10):
        self.lambda_gb = lambda_gb
    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate discriminator scores
        mixed_scores = discriminator(interpolated_images)
    
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
class WassersteinDistance():
    def __init__(self, scale=0.001):
        self.scale = scale

    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (-(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                    + self.scale * torch.mean(discriminator_real ** 2))
    
class HingeLoss():
    def __init__(self):
        self.relu = torch.nn.ReLU()

    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (torch.mean(self.relu(1.0 - discriminator_real))
                    + torch.mean(self.relu(1.0 + discriminator_fake)))

class LeastSquaresGANLoss():
    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (torch.mean((discriminator_real - 1.0) ** 2)
                    + torch.mean(discriminator_fake ** 2))
    
class StandardGANLoss():
    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (torch.mean(torch.log(discriminator_real))
                    + torch.mean(torch.log(1.0 - discriminator_fake)))
    
class StandardGANLossWithSigmoid():
    def __init__(self):
        self.sigmoid = torch.nn.Sigmoid()

    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (torch.mean(self.sigmoid(discriminator_real))
                    + torch.mean(self.sigmoid(1.0 - discriminator_fake)))
    
class StandardGANLossWithLogSigmoid():
    def __init__(self):
        self.logsigmoid = torch.nn.LogSigmoid()

    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (torch.mean(self.logsigmoid(discriminator_real))
                    + torch.mean(self.logsigmoid(1.0 - discriminator_fake)))
    
class StandardGANLossWithBCE():
    def __init__(self):
        self.bce = torch.nn.BCELoss()

    def __call__(self, real, fake, discriminator, discriminator_real, discriminator_fake, device):
        return (self.bce(discriminator_real, torch.ones_like(discriminator_real))
                    + self.bce(discriminator_fake, torch.zeros_like(discriminator_fake)))
