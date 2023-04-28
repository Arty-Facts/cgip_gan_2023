from utils.setup import setup_experiment
from multiprocessing import freeze_support
import argparse
import torch
from utils.utils import load_from_yaml

def main(config_file):
    config = load_from_yaml(config_file)
    experiment = setup_experiment(config)
    # torch.autograd.set_detect_anomaly(True)
    experiment.fit()

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
