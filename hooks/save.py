from utils.utils import save_to_json, save_to_yaml
import torch

SAVE = {
    ".json": save_to_json,
    ".yaml": save_to_yaml,
    ".pkl": torch.save,
} 

class Save:
    def __init__(self, supervisor, targets, every, dir):
        self.targets = targets 
        self.every = every
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.epoch_end_done = False

    def ping(self):
        assert self.epoch_end_done

    def epoch_end(self):
        self.epoch_end_done = True
        epochs = self.supervisor.meta["epochs"]

        if epochs % self.every == 0:
            for name, target in self.targets.items():
                path = self.path / target
                item = self.supervisor[name]
                if isinstance(item, torch.nn.DataParallel):
                    data = item.module.state_dict()
                if isinstance(item, dict):
                    data = item
                else:
                    data = item.state_dict()
                SAVE[path.suffix](data, path)
