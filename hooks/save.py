from utils.utils import save_to_json, save_to_yaml
import torch, logging

SAVE = {
    ".json": save_to_json,
    ".yaml": save_to_yaml,
    ".pkl": torch.save,
} 

class Save:
    def __init__(self, supervisor, targets, every, dir, save_all=True):
        self.targets = targets 
        self.every = every
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.save_all = save_all
        self.path.mkdir(parents=True, exist_ok=True)
        self.epoch_end_done = False

    def ping(self):
        assert self.epoch_end_done

    def epoch_end(self):
        self.epoch_end_done = True
        epochs = self.supervisor.meta["epochs"]
        images = self.supervisor.meta["images"]

        if epochs % self.every == 0:
            if self.save_all:
                model_path = self.supervisor.base_path / "history" / f"{images}"
                model_path.mkdir(parents=True, exist_ok=True)
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
                logging.info(f"Saved {name} to {path}")
                if self.save_all:
                    path = model_path / target
                    SAVE[path.suffix](data, path)
                    logging.info(f"Saved {name} to {path}")
