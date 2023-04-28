import importlib
import torchvision
from typing import Any, Union
from utils.utils import save_to_yaml
from hooks.hook_handler import HookHandler


class ValueAccumulator:
    def __init__(self, scales_and_funs):
        self.scales_and_funs = scales_and_funs
        self.name = "+".join(map(lambda sf: f"{sf[0]}*{sf[1].__class__.__name__}", sorted(self.scales_and_funs, key=lambda sf:sf[1].__class__.__name__)))
    def __repr__(self) -> str:
        return self.name
    def __call__(self, yhats, ys):
        score = sum(map(lambda sf: sf[0]*sf[1](yhats, ys), self.scales_and_funs))
        return score

def load_module(module, func, *args, **kvargs):
    mod = importlib.import_module(module)
    func = getattr(mod, func)
    return func(*args, **kvargs)

def setup_module(par, **kvargs) -> Any:
    if isinstance(par, dict):
        module, name, args = par["module"], par["name"], par["args"]
    elif isinstance(par, (list, tuple)) and len(par) == 3:
        module, name, args = par
    else:
        raise ValueError(f"Not supported module loading for {par}")
    return load_module(module, name, **args, **kvargs)

def setup_transforms(par) -> Union[Any, None]:
    if par is None:
        return None
    return torchvision.transforms.Compose(
        [load_module(module, name, **args) for module, name, args in par]
    )

def setup_dataset(par, transform) -> Any:
    return load_module(par["module"], par["name"], transform=transform, **par["args"])

def setup_dataloader(par, dataset) -> Any:
    return load_module(par["module"], par["name"], dataset, **par["args"])

def setup_data(supervisor, par) -> None:
    for name, conf in par.items():
        transforms = setup_transforms(conf.get("transforms", None))
        dataset = setup_dataset(conf["dataset"], transforms)
        dataloader = setup_dataloader(conf["dataloader"], dataset)
        supervisor[name] = dataloader

def setup_models(supervisor, par) -> None:
    for name, conf in par.items():
        model = setup_module(conf).to(supervisor.target_device)
        supervisor[name] = model

def setup_optimizer(supervisor, par):
    module, name, target, args = par["module"], par["name"], par["target"], par["args"]
    model = supervisor[target]
    return load_module(module, name, model.parameters(), **args)

def setup_optimizers(supervisor, par) -> None:
    for name, conf in par.items():
        model = setup_optimizer(supervisor, conf)
        supervisor[name] = model

def setup_loss(supervisor, par):
    if par != None:
        for name, losses in par.items():
            acc_loss = []
            for scale, module, name, args in losses: 
                fun = load_module(module, name, **args)
                acc_loss.append((scale, fun))
            loss = ValueAccumulator(acc_loss)
            supervisor[name] = loss

def setup_score(supervisor, par) -> None:
    score = []
    for scale, module, name, args in par: 
        fun = load_module(module, name, **args)
        score.append((scale, fun))
    score = ValueAccumulator(score)
    supervisor["score"] = score

def setup_hooks(supervisor, par):
    hook_handler = HookHandler()
    for conf in par:
        hook_handler.register(setup_module(conf, supervisor=supervisor))
    supervisor["hooks"] = hook_handler

def setup_trainer(supervisor, par) -> None:
    module, name, args = par["module"], par["name"], par["args"]
    trainer =  load_module(module, name, supervisor, **args)
    supervisor["trainer"] = trainer


def setup_experiment(config):
    supervisor = setup_module(config['supervisor'])
    setup_data(supervisor, config["data"])
    setup_models(supervisor, config["models"])
    setup_optimizers(supervisor, config["optimizers"])

    setup_loss(supervisor, config["loss"])
    setup_hooks(supervisor, config["hooks"])
    setup_trainer(supervisor, config["trainer"])
    # setup_score(supervisor, config["score"])

    save_to_yaml(config, supervisor.base_path / 'config.yaml')
    
    return supervisor
