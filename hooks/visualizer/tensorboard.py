from torch.utils.tensorboard import SummaryWriter
import numpy, torch

class Tensorboard:
    def __init__(self, base_dir=".", out_dir="visualizer") -> None:
        self.writer = SummaryWriter(log_dir=f"{base_dir}/{out_dir}")
        
    def add_scalar(self, name, value, step):
        if isinstance(value, (int, float, bool)):
            self.writer.add_scalar(name, value, step)
        else:
            print(f"Cant send {type(value)} to tensorboard")
        
    def add_histogram(self, name, values, step, **kvargs):
        if isinstance(values, (tuple, list, numpy.ndarray, torch.Tensor)):
            self.writer.add_histogram(name, values, step, **kvargs)
        else:
            print(f"Cant send {type(values)} to tensorboard")
