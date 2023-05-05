
from metrics.utils import activation_statistics, frechet_distance, mean_var_cosine_similarity, inception_activation_statistics
import torch
from torch.utils.data import random_split
from metrics.inception import InceptionV3

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

        real_images = [torch.unsqueeze(subset[i][0], 0) for i in range(self.samples)]
        
        mean_real, std_real = activation_statistics(real_images, encoder, device)
        mean_fake, std_fake = activation_statistics(fake_images, encoder, device)

        fd = frechet_distance(mean_real, std_real, mean_fake, std_fake)
        return fd

class FID:
    """Latent Frechet Distance Score"""
    def __init__(self, supervisor, generator, data, dims=2048, samples=100):
        self.supervisor = supervisor
        self.generator = generator
        self.samples = samples
        self.data = data
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        device = self.supervisor.target_device
        self.model = InceptionV3([block_idx]).to(device)
        data = self.supervisor[self.data].dataset
        real_images = [torch.unsqueeze(data[i][0], 0) for i in range(len(data))]
        self.mean_real, self.std_real = inception_activation_statistics(real_images, self.model, device)


    def __call__(self):
        generator = self.supervisor[self.generator]
        device = self.supervisor.target_device
        fake_images = [generator(torch.randn(1, generator.z_dim, device=device)) for _ in range(self.samples)]
        mean_fake, std_fake = inception_activation_statistics(fake_images, self.model, device)
        fd = frechet_distance(self.mean_real, self.std_real, mean_fake, std_fake)
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

        real_images = [torch.unsqueeze(avg_pool(subset[i][0].to(device)), 0) for i in range(self.samples)]

        mean_cos, var_cos, stack_cos, cov_cos, vmr_cos = mean_var_cosine_similarity(real_images, fake_images)
        return 1-abs(mean_cos), 1-abs(var_cos), 1-abs(stack_cos), 1-abs(cov_cos), 1-abs(vmr_cos)
