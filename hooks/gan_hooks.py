from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

class GenerateTensorboardImageSample:
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
        self.writer = SummaryWriter(self.supervisor.base_path / self.__class__.__name__)
        self.fixed_noise = torch.randn(self.samples, self.supervisor[self.target_model].w_dim, device=self.device)

    def epoch_end(self):
        self.epoch_end_done = True
        # Print losses occasionally and print to tensorboard
        if self.supervisor.meta["epochs"] % self.every == 0:
            with torch.no_grad():
                fake = self.supervisor[self.target_model](self.fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                self.writer.add_image(self.target_model, img_grid_fake, global_step=self.supervisor.meta["epochs"])

class ConsoleStats:
    def __init__(self, supervisor, every=100):
        self.every = every
        self.supervisor = supervisor


    def batch_end(self):
        c = self.supervisor.cache
        m = self.supervisor.meta

        if self.supervisor.meta["steps"] % self.every == 0:
            loss_gen = c.loss_gen.item()
            loss_disc = c.loss_disc.item()
            print(f'[{m["current_epochs"]}/{m["end_epochs"]}]\Loss_disc: {loss_disc:.4f}\tLoss_G: {loss_gen:.4f}')