from utils.setup import setup_experiment
from multiprocessing import freeze_support
import argparse, torch, logging
from utils.utils import load_from_yaml, update_dict

def main(config_file, updates=None):
    config = load_from_yaml(config_file)
    if updates is not None:
        update_dict(config, updates)
    experiment = setup_experiment(config)
    # torch.autograd.set_detect_anomaly(True)
    experiment.fit()

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to the config file")
    parser.add_argument("-u", "--update", nargs='+',
                        help="update config file")
    args = parser.parse_args()
    # logging.DEBUG,
    # logging.INFO,
    # logging.WARNING,
    # logging.ERROR,
    # logging.CRITICAL
    logging.basicConfig(level=logging.WARNING)
    # logging.disable(logging.DEBUG)
    main(args.config, updates=args.update)
