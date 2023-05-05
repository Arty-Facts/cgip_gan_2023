from pathlib import Path
from utils.utils import load_from_json, load_from_yaml
import torch, logging

LOAD = {
    ".json": load_from_json,
    ".yaml": load_from_yaml,
    ".pkl": torch.load,
} 

class Load:
    def __init__(self, supervisor, targets, dir):
        self.targets = targets 
        self.supervisor = supervisor
        self.path = Path(dir)
        self.init_done = False

    def ping(self):
        assert self.init_done

    def init(self):
        self.init_done = True
        for name, target in self.targets.items():
            path = self.path / target
            if path.exists():
                self.supervisor[name].load_state_dict(LOAD[path.suffix](path))
                logging.debug(f"Loading {name} from {path}")
            else:
                logging.warning(f"Did not find {path} starting {name} from scratch")

class Resume:
    def __init__(self, supervisor, targets, dir):
        self.targets = targets 
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.init_done = False

    def ping(self):
        assert self.init_done

    def init(self):
        self.init_done = True
        for name, target in self.targets.items():
            path = self.path / target
            if path.exists():
                self.supervisor[name].load_state_dict(LOAD[path.suffix](path))
                logging.debug(f"Resuming {name} from {path}")
            else:
                logging.warning(f"Did not find {path} starting {name} from scratch")