from datetime import datetime
from utils.utils import run_cmd
import logging

class MetaData:
    def __init__(self) -> None:
        now = datetime.now()
        self.data = {
            "steps": 0,
            "epochs": 0,
            "images": 0,
            "start_time": now.strftime("%Y/%m/%d %H:%M:%S"),
            "git-hash" : run_cmd("git rev-parse HEAD").strip() 
        }

    def __getitem__(self, name: str):
        return self.data[name]
        
    def __setitem__(self, name: str, value) -> None:
        self.data[name] = value

    def increment(self, name, value=1):
        if name not in self.data:
            self.data[name] = 0
        self.data[name] += value
        return self.data[name] 

    def append(self, name, value):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((self.data["images"], value))

    def get_last(self, name):
        if name not in self.data:
            logging.warning(f"Item {name} not found in meta data")
            return None, float("nan")
        if not isinstance(self.data[name], (list, tuple)):
            logging.warning(f"Item {name} is not a collection")
            return None, float("nan")
        return self.data[name][-1]
    
    def keys(self):
        return self.data.keys()
    
    def state_dict(self):
        return self.data
    
    def load_state_dict(self, data):
        self.data = data

