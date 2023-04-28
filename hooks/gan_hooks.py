from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

class TensorboardGenerateImageSample:
    def __init__(self, supervisor, target_model=None, samples=32, every=100):
        self.target_model = target_model 
        self.samples = samples
        self.every = every

        self.start_done = False
        self.epoch_end_done = False
        self.supervisor = supervisor

    def ping(self):
        assert self.start_done
        assert self.epoch_end_done

    def start(self):
        self.start_done = True
        self.device = self.supervisor.target_device
        self.writer = SummaryWriter(self.supervisor.base_path)
        self.fixed_noises = [torch.randn(1, self.supervisor[self.target_model].z_dim, device=self.device) for _ in range(self.samples)]

    def epoch_end(self):
        self.epoch_end_done = True
        # Print losses occasionally and print to tensorboard
        if self.supervisor.meta["epochs"] % self.every == 0:
            with torch.no_grad():
                fake = torch.cat([self.supervisor[self.target_model](fixed_noise).detach().cpu() for fixed_noise in self.fixed_noises])
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                self.writer.add_image(self.target_model, img_grid_fake, global_step=self.supervisor.meta["epochs"])

class TensorboardScalarData:
    def __init__(self, supervisor, targets, name="Scalars", every=100):
        self.every = every
        self.supervisor = supervisor
        self.targets = targets
        self.name = name
        self.writer = SummaryWriter(self.supervisor.base_path)

    def batch_end(self):
        s = self.supervisor

        if s.meta["steps"] % self.every == 0:
            for target in self.targets:
                self.writer.add_scalar(f"{self.name}/{target}", float(s[target]), global_step=s.meta["steps"])

class ConsoleStats:
    def __init__(self, supervisor, targets, every=100):
        self.every = every
        self.supervisor = supervisor
        self.targets = targets


    def batch_end(self):
        s = self.supervisor
        steps_per_epoch = len(s["data"])

        if s.meta["steps"] % self.every == 0:
            data = []
            for target in self.targets:
                data.append(f'{target}: {float(s[target]):7.4f}')
            print(f'[{s.meta["epochs"]}/{s.meta["end_epochs"]}][{s.meta["steps"]}/{s.meta["end_epochs"]*steps_per_epoch}] '+', '.join(data))