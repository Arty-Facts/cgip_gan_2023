from pathlib import Path
from utils.utils import load_from_json, load_from_yaml
import torch

LOAD = {
    ".json": load_from_json,
    ".yaml": load_from_yaml,
    ".pkl": torch.load,
} 

class Resume:
    def __init__(self, supervisor, targets, dir):
        self.targets = targets 
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.start_done = False

    def ping(self):
        assert self.start_done

    def start(self):
        self.start_done = True
        for name, target in self.targets.items():
            path = self.path / target
            if path.exists():
                self.supervisor[name].load_state_dict(LOAD[path.suffix](path, map_location=self.supervisor.target_device))
                print(f"Loading {name} from {path}")
            else:
                print(f"Did not find {path} starting {name} from scratch")

class Resume:
    def __init__(self, supervisor, targets, dir):
        self.targets = targets 
        self.supervisor = supervisor
        self.path = supervisor.base_path / dir
        self.start_done = False

    def ping(self):
        assert self.start_done

    def start(self):
        self.start_done = True
        for name, target in self.targets.items():
            path = self.path / target
            if path.exists():
                self.supervisor[name].load_state_dict(LOAD[path.suffix](path, map_location=self.supervisor.target_device))
                print(f"Loading {name} from {path}")
            else:
                print(f"Did not find {path} starting {name} from scratch")