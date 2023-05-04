from pathlib import Path
from typing import Any
from supervisors.meta_data import MetaData

class Cache: pass

class Supervisor:
    def __init__(
            self,
            name = None,
            version = 0,
            seed = 1337,
            device = "cuda",
            nodes = [0], 
            checkpoint = 'checkpoints',
        ) -> None:
        
        self.meta = MetaData()
        self.base_path = Path(f"{checkpoint}/{name}/{version}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.meta["name"] = name
        self.meta["version"] = version
        self.meta["seed"] = seed
        self.device = device
        self.nodes = nodes
        if device == "cuda" and len(nodes) == 1:
            self.target_device = f"{device}:{nodes[0]}"        
        else:
            self.target_device = device
        self.checkpoint = checkpoint
        self.cache = Cache() # cache to store intermediate values
        self.register = {"meta": self.meta}

    
    def __getitem__(self, name: str) -> Any:
        """Get an item from the register or cache"""
        if name in self.register:
            return self.register[name]
        if hasattr(self.cache, name):
            return getattr(self.cache, name)
        
    def __setitem__(self, name: str, value: Any) -> None:
        self.register[name] = value

    def fit(self):
        self.register["trainer"]()