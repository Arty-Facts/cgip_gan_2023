from typing import Any
from utils.setup import setup_module
from utils.utils import set_seed
import torch

def gradient_penalty(discriminator, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = discriminator(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(#Computes and returns the sum of gradients of outputs with respect to the inputs(dl/di?)
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)#||a||
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class StyleGanTrainer:
    def __init__(
            self,
            supervisor,
            epochs,
            data,
            generator, 
            discriminator, 
            optimizer_generator, 
            optimizer_discriminator,
            generator_loss,
            discriminator_loss,
        ) -> None:
        self.supervisor = supervisor
        self.data = data
        self.generator = generator 
        self.discriminator = discriminator 
        self.optimizer_generator = optimizer_generator 
        self.optimizer_discriminator = optimizer_discriminator 
        self.supervisor.meta["current_epochs"] = 0
        self.supervisor.meta["end_epochs"] = epochs
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def __call__(self) -> Any:
        c = self.supervisor.cache
        c.data = self.supervisor[self.data]
        hooks = self.supervisor["hooks"]
        device = self.supervisor.target_device
        set_seed(self.supervisor.meta["seed"])
        
        hooks.call("start")
        for epoch in range(self.supervisor.meta["current_epochs"], self.supervisor.meta["end_epochs"]):
            self.supervisor.meta["current_epochs"] = epoch
            hooks.call("epoch_begin")
            netD = self.supervisor[self.discriminator]
            netG = self.supervisor[self.generator]
            lossG = self.supervisor[self.generator_loss]
            lossD = self.supervisor[self.discriminator_loss]
            optD = self.supervisor[self.optimizer_discriminator]
            optG = self.supervisor[self.optimizer_generator]
            for real, labels in c.data:
                hooks.call("batch_start")
                c.labels = labels.to(device)
                c.real = real.to(device)

                c.cur_batch_size = c.real.shape[0]
                c.noise = torch.randn(c.cur_batch_size, netG.z_dim, device=device)

                c.fake  = netG(c.noise)
                c.d_real = netD(c.real)
                c.d_fake = netD(c.fake)
                c.loss_disc = lossD(c.real, c.fake, netD, c.d_real, c.d_fake, device)
                optD.zero_grad()
                c.loss_disc.backward(retain_graph=True)
                optD.step() #Update the discriminator model parameters

                c.loss_gen = lossG(c.fake, netD, device)

                netG.zero_grad()
                c.loss_gen.backward()
                optG.step()

                self.supervisor.meta["steps"] += 1
                hooks.call("batch_end")
            self.supervisor.meta["epochs"] += 1

            hooks.call("epoch_end")
            hooks.call("ping")
        hooks.call("end")
        
        