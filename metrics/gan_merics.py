
from metrics.utils import activation_statistics, frechet_distance, mean_var_cosine_similarity
import torch
from torch.utils.data import random_split

class LFDS:
    """Latent Frechet Distance Score"""
    def __init__(self, supervisor, encoder, generator, data, samples=100):
        self.supervisor = supervisor
        self.encoder = encoder
        self.generator = generator
        self.samples = samples
        self.data = data

    def __call__(self):
        generator = self.supervisor[self.generator]
        data = self.supervisor[self.data].dataset
        encoder = self.supervisor[self.encoder]
        device = self.supervisor.target_device

        fake_images = [generator(torch.randn(1, generator.z_dim, device=device)) for _ in range(self.samples)]
        subset, _ = random_split(data, [self.samples, len(data) - self.samples])

        real_images = [subset[i][0] for i in range(self.samples)]
        
        mean_real, std_real = activation_statistics(real_images, encoder, device)
        mean_fake, std_fake = activation_statistics(fake_images, encoder, device)

        fd = frechet_distance(mean_real, std_real, mean_fake, std_fake)
        return fd


class StatsCosineSimilarity:
    """Image Mean Var Cosine Similarity"""
    def __init__(self, supervisor, generator, data, samples=100):
        self.supervisor = supervisor
        self.generator = generator
        self.samples = samples
        self.data = data

    def __call__(self):
        generator = self.supervisor[self.generator]
        data = self.supervisor[self.data].dataset
        device = self.supervisor.target_device

        # avg_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        avg_pool = torch.nn.Identity()

        fake_images = [avg_pool(generator(torch.randn(1, generator.z_dim, device=device))) for _ in range(self.samples)]
        subset, _ = random_split(data, [self.samples, len(data) - self.samples])

        real_images = [avg_pool(subset[i][0].to(device)) for i in range(self.samples)]

        mean_cos, var_cos, stack_cos, cov_cos, vmr_cos = mean_var_cosine_similarity(real_images, fake_images)
        return 1-abs(mean_cos), 1-abs(var_cos), 1-abs(stack_cos), 1-abs(cov_cos), 1-abs(vmr_cos)
