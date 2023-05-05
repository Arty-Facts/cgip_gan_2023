
import subprocess, yaml, json, numpy, torch, random, os, logging


def run_cmd(cmd):
    """Run a command and return the output"""
    process = subprocess.run(cmd.split(" "), universal_newlines=True, stdout=subprocess.PIPE)
    stdout = process.stdout
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout)
    return stdout


def save_to_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, encoding="utf-8", default_flow_style=None, sort_keys=False)
        
def load_from_yaml(path, map_location=None):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=False, indent=4)
        
def load_from_json(path, map_location=None):
    with open(path, "r") as f:
        return json.load(f)
    

def set_seed(seed: int = 1337) -> None:
    """Set the random seed for reproducibility"""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")

def update_dict(container, updates):
    for update in updates:
        path, value = update.split('=')
        path = path.split('.')
        config = update_dict_recursively(path, value, container)
        logging.info(f"Updated {path} to {value}")
    return config

def update_dict_recursively(path, value, container):
    """Update a dictionary recursively"""
    if len(path) == 1:
        container[path[0]] = yaml.safe_load(value)
        return container
    if path[0] not in container:
        container[path[0]] = {}
        return update_dict_recursively(path[1:], value, container[path[0]])
    return update_dict_recursively(path[1:], value, container[path[0]])