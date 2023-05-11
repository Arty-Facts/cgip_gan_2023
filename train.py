from utils.setup import setup_experiment
import argparse, torch, logging, time, multiprocessing
from utils.utils import load_from_yaml, update_dict, optuna_traning_config
from pathlib import Path
from functools import partial
from collections import defaultdict

import optuna

def schedul_study(scheduler_config, study, fun, config, parameters, optimize_config, updates, trials):
    running_processes = defaultdict(list)
    trials_left = trials
    while trials_left > 0:
        for name, conf in scheduler_config.items():
            logging.info(f"Running scheduler {name}")
            device = conf["device"]
            node = conf["node"]
            processes = conf["processes"]
            if device == "cuda":
                device = f"{device}:{node}"
            if updates is None:
                updates = []
            updates.append(f"supervisor.args.device={device}")
            updates.append(f"supervisor.args.nodes=[{device}]")
            logging.debug(f"Device: {device} Node: {node} Processes: {processes}")
            objective = partial(fun, config, parameters, optimize_config, updates)
            
            while trials_left > 0 and len(running_processes[name]) <= processes:
                p = multiprocessing.Process(target=study.optimize, args=(objective, ), kwargs={"n_trials": 1})
                p.start()
                running_processes[name].append(p)
                trials_left -= 1
        for name, active_processes in running_processes.items():
            for process in active_processes:
                if not process.is_alive():
                    running_processes[name].remove(process)
        time.sleep(0.1)  # Adjust as needed to control the frequency of checking
        
    for name, active_processes in running_processes.items():
        for process in active_processes:
            process.join() # Wait for all processes to finish


def train(config, updates=None):
    if updates is not None:
        update_dict(config, updates)
    experiment = setup_experiment(config)
    # torch.autograd.set_detect_anomaly(True)
    return experiment.fit()

def train_plan(config, training_config, updates=None):
    score = None
    for name, conf in training_config.items():
        logging.info(f"Training {name}")
        update_dict(config, conf)
        score = train(config, updates)
        logging.info(f"Score: {score}")
    return score

def train_opt(config, optimize_config, scheduler=None, updates=None):
    parameters = optimize_config.pop("parameters")
    trials = int(optimize_config.pop("trials"))
    study_name = optimize_config.pop("name")
    db = optimize_config.pop("db")
    def objective(config, parameters, optimize_config, updates, trial):
        training_conf = optuna_traning_config(optimize_config, parameters, trial)
        if updates is None:
            updates = []
        updates.append(f"supervisor.args.version={trial.number}")
        logging.info(f"Trial {trial.number}/{trials}")
        logging.debug(f"Trial config: {training_conf}")
        return train_plan(config, training_conf, updates)
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{db}"
    logging.info(f"Optuna storage: {storage_name}")
    logging.info(f"Inspect using optuna-dashboard {storage_name}")
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)
    if scheduler is None:
        study.optimize(partial(objective, config, parameters, optimize_config, updates), n_trials=trials)
    else:
        scheduler_config = load_from_yaml(scheduler)
        schedul_study(scheduler_config, study, objective, config, parameters, optimize_config, updates, trials)
    return study.best_value

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-t", "--training-config", type=str, default=None, help="training config file")
    parser.add_argument("-u", "--update", nargs='+', default=None, help="update config file")
    parser.add_argument("-o", "--optimize", default=None, help="optimize config file for hyperparameter search")
    parser.add_argument("-s", "--scheduler", default=None, help="device scheduler config file")
    args = parser.parse_args()
    # logging.DEBUG,
    # logging.INFO,
    # logging.WARNING,
    # logging.ERROR,
    # logging.CRITICAL
    logging.basicConfig(level=logging.INFO)
    # logging.disable(logging.DEBUG)
    config = load_from_yaml(args.config)
    if args.optimize is not None and args.training_config is not None:
        print("Optimization and training config cannot be used together")
        exit(1)

    if args.training_config is None and args.optimize is None:
        score = train(config, updates=args.update)
    elif  args.optimize is not None:
        optimize_config = load_from_yaml(args.optimize)
        score = train_opt(config, optimize_config, updates=args.update)
    else:
        train_config = load_from_yaml(args.training_config)
        score = train_plan(config, train_config, scheduler=args.scheduler, updates=args.update)
    logging.info(f"Final Score: {score}")
