from utils.utils import save_to_json, save_to_yaml
import torch, logging

SAVE = {
    ".json": save_to_json,
    ".yaml": save_to_yaml,
    ".pkl": torch.save,
} 

class SaveToDir:
    def __init__(self, supervisor, targets, every, dir):
        self.targets = targets 
        self.every = every
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.epoch_end_done = False

    def ping(self):
        assert self.epoch_end_done

    def _save(self):
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
                logging.debug(f"Saved {name} to {path}")


    def epoch_end(self):
        self.epoch_end_done = True
        epochs = self.supervisor.meta["epochs"]

        if epochs % self.every == 0:
            self._save()
            
    def end(self):
        self._save()

class SaveAll:
    def __init__(self, supervisor, targets, every):
        self.targets = targets 
        self.every = every
        self.supervisor = supervisor
        self.epoch_end_done = False

    def ping(self):
        assert self.epoch_end_done

    def _save(self):
        images = self.supervisor.meta["images"]
        model_path = self.supervisor.base_path / "history" / f"{images}"
        model_path.mkdir(parents=True, exist_ok=True)
        for name, target in self.targets.items():
            item = self.supervisor[name]
            if isinstance(item, torch.nn.DataParallel):
                data = item.module.state_dict()
            if isinstance(item, dict):
                data = item
            else:
                data = item.state_dict()
            path = model_path / target
            SAVE[path.suffix](data, path)
            logging.debug(f"Saved {name} to {path}")

    def epoch_end(self):
        self.epoch_end_done = True
        epochs = self.supervisor.meta["epochs"]

        if epochs % self.every == 0:
            self._save()
            

    def end(self):
        self._save()

class SaveBest:
    def __init__(self, supervisor, targets, every):
        self.targets = targets 
        self.every = every
        self.supervisor = supervisor
        self.path = supervisor.base_path / "best"
        self.path.mkdir(parents=True, exist_ok=True)
        self.best_score = None
        self.epoch_end_done = False

    def ping(self):
        assert self.epoch_end_done

    def _save(self):
        score = self.supervisor["score"](self.supervisor.meta)
        if self.best_score is None:
            self.best_score = score
        if score <= self.best_score:
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
                logging.debug(f"Saved {name} to {path}")
            self.best_score = score

    def epoch_end(self):
        self.epoch_end_done = True
        epochs = self.supervisor.meta["epochs"]

        if epochs % self.every == 0:
            self._save()

    def end(self):
        self._save()
