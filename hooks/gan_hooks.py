from torch.utils.tensorboard import SummaryWriter
from metrics.gan_merics import LFDS, StatsCosineSimilarity, FID
import torchvision
import torch
from tqdm import tqdm

class TensorboardGenerateImageSample:
    def __init__(self, supervisor, target_model=None, samples=32, global_steps="images", every=100):
        self.target_model = target_model 
        self.samples = samples
        self.every = every

        self.start_done = False
        self.epoch_end_done = False
        self.supervisor = supervisor
        self.global_steps = global_steps

    def ping(self):
        assert self.start_done
        assert self.epoch_end_done

    def start(self):
        self.start_done = True
        self.device = self.supervisor.target_device
        self.writer = SummaryWriter(self.supervisor.base_path)

        self.image_path = self.supervisor.base_path / "gen_images"
        self.image_path.mkdir(parents=True, exist_ok=True)

        self.fixed_noises = [torch.randn(1, self.supervisor[self.target_model].z_dim, device=self.device) for _ in range(self.samples)]

    def epoch_end(self):
        self.epoch_end_done = True
        s = self.supervisor
        # Print losses occasionally and print to tensorboard
        if s.meta["epochs"] % self.every == 0:
            curr_path = self.image_path / str(s.meta[self.global_steps])
            curr_path.mkdir(parents=True, exist_ok=True)
            scale = lambda x: x*0.5 + 0.5 

            with torch.no_grad():
                fake = [scale(s[self.target_model](fixed_noise).detach().cpu()) for fixed_noise in self.fixed_noises]
                for i, img in enumerate(fake):
                    torchvision.utils.save_image(img, curr_path / f"{i}.png")
                img_grid_fake = torchvision.utils.make_grid(torch.cat(fake), normalize=True)
                self.writer.add_image(self.target_model, img_grid_fake, global_step=s.meta[self.global_steps])


class TensorboardLFDS:
    def __init__(self, supervisor, encoder, generator, data, samples=100, name="LFDS", global_steps="images",  every=100):
        self.lfds = LFDS(supervisor, encoder, generator, data, samples)
        self.generator = generator
        self.every = every
        self.supervisor = supervisor
        self.name = name
        self.global_steps = global_steps
        self.writer = SummaryWriter(self.supervisor.base_path)

    def epoch_end(self):
        s = self.supervisor
        if s.meta["epochs"] % self.every == 0:
            score = self.lfds()
            s.meta.append(f"{self.name}", float(score))
            self.writer.add_scalar(f"{self.name}", float(score), global_step=s.meta[self.global_steps])

class TensorboardFID:
    def __init__(self, supervisor, generator, data, dim=2048, samples=100, name="FID", global_steps="images",  every=100):
        self.generator = generator
        self.every = every
        self.supervisor = supervisor
        self.data = data
        self.dim = dim
        self.samples = samples
        self.name = name
        self.global_steps = global_steps
        self.writer = SummaryWriter(self.supervisor.base_path)

    def start(self):
        self.fid = FID(self.supervisor, self.generator, self.data, self.dim, self.samples)

    def epoch_end(self):
        s = self.supervisor
        if s.meta["epochs"] % self.every == 0:
            score = self.fid()
            s.meta.append(f"{self.name}", float(score))
            self.writer.add_scalar(f"{self.name}", float(score), global_step=s.meta[self.global_steps])

class TensorboardImageStatsCosineSimilarity:
    def __init__(self, supervisor, generator, data, samples=100, name="StatsCosineSimilarity", global_steps="images",  every=100):
        self.imvcs = StatsCosineSimilarity(supervisor, generator, data, samples)
        self.generator = generator
        self.every = every
        self.supervisor = supervisor
        self.name = name
        self.global_steps = global_steps
        self.writer = SummaryWriter(self.supervisor.base_path)

    def epoch_end(self):
        s = self.supervisor
        if s.meta["epochs"] % self.every == 0:
            mean_cos, var_cos, stack_cos, cov_cos, vmr_cos = self.imvcs()
            s.meta.append(f"{self.name}/mean", float(mean_cos))
            s.meta.append(f"{self.name}/var", float(var_cos))
            s.meta.append(f"{self.name}/stack", float(stack_cos))
            s.meta.append(f"{self.name}/cov", float(cov_cos))
            s.meta.append(f"{self.name}/vmr", float(vmr_cos))
            self.writer.add_scalar(f"{self.name}/mean", float(mean_cos), global_step=s.meta[self.global_steps])
            self.writer.add_scalar(f"{self.name}/var", float(var_cos), global_step=s.meta[self.global_steps])
            self.writer.add_scalar(f"{self.name}/stack", float(stack_cos), global_step=s.meta[self.global_steps])
            self.writer.add_scalar(f"{self.name}/cov", float(cov_cos), global_step=s.meta[self.global_steps])
            self.writer.add_scalar(f"{self.name}/vmr", float(vmr_cos), global_step=s.meta[self.global_steps])


class TensorboardScalarData:
    def __init__(self, supervisor, targets, name="Scalars", global_steps="images",  every=100):
        self.every = every
        self.supervisor = supervisor
        self.targets = targets
        self.name = name
        self.global_steps = global_steps
        self.writer = SummaryWriter(self.supervisor.base_path)

    def batch_end(self):
        s = self.supervisor
        if s.meta["steps"] % self.every == 0:
            for target in self.targets:
                self.writer.add_scalar(f"{self.name}/{target}", float(s[target]), global_step=s.meta[self.global_steps])

class ConsoleStats:
    def __init__(self, supervisor, targets, every=100):
        self.every = every
        self.supervisor = supervisor
        self.targets = targets

    def start(self):
        s = self.supervisor
        self.bar = tqdm(total=s.meta["end_epochs"], initial=s.meta["epochs"])
        self.bar.set_description(f'Epochs')

    def epoch_end(self):
        self.bar.update()

    def batch_end(self):
        s = self.supervisor

        if s.meta["steps"] % self.every == 0:
            for target in self.targets:
                s.meta.append(target, float(s[target]))
            self.bar.set_postfix({target: f'{float(s[target]):7.4f}' for target in self.targets})

class SaveToRegister:
    def __init__(self, supervisor, targets, every=100):
        self.every = every
        self.supervisor = supervisor
        self.targets = targets

    def batch_end(self):
        s = self.supervisor

        if s.meta["steps"] % self.every == 0:
             for target in self.targets:
                s.meta.append(target, float(s[target]))


class UnfreezLayers:
    def __init__(self, supervisor, targets, after=100):
        self.after = after
        self.supervisor = supervisor
        self.targets = targets
        self.done = False

    def start(self):
        s = self.supervisor
        for target in self.targets:
            for layer in s[target].layers:
                if layer.pretrained:
                    for param in layer.parameters():
                        param.requires_grad = False
                    
    def epoch_end(self):
        if self.done:
            return
        s = self.supervisor

        if s.meta["epochs"] == self.after:
            for target in self.targets:
                for layer in s[target].layers:
                    if layer.pretrained:
                        for param in layer.parameters():
                            param.requires_grad = True
        self.done = True