from utils.setup import setup_experiment
from multiprocessing import freeze_support
import argparse, torch, logging
from utils.utils import load_from_yaml, update_dict, optuna_traing_conf
import optuna

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

def train_opt(config, optimize_config, updates=None):
    parameters = optimize_config.pop("parameters")
    trials = int(optimize_config.pop("trials"))
    db = optimize_config.pop("db")
    def objective(trial):
        training_conf = optuna_traning_config(optimize_config, parameters, trial)
        return train_plan(config, training_conf, updates)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, storage=db)
    return study.best_value

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-t", "--training-config", type=str, default=None, help="training config file")
    parser.add_argument("-u", "--update", nargs='+', default=None, help="update config file")
    parser.add_argument("-o", "--optimize", default=None, help="optimize config file for hyperparameter search")
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
        score = train_plan(config, train_config, updates=args.update)
    logging.info(f"Final Score: {score}")
