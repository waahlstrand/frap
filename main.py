import argparse
import os
import torch
import utils

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default='data', help='Path to the data.')
parser.add_argument('--config_dir', default='config', help="Directory containing config.json")


if __name__ == '__main__':

    # Parse arguments to program
    args = parser.parse_args()

    data_path   = args.data_path
    config_dir  = args.config_dir

    config_path = os.path.join(config_dir + "config.json")

    # Extract parameters
    config = utils.Params(config_path)

    # Set manual random seed
    if config.cuda: # If cuda
        torch.cuda.manual_seed(222)
    else:
        torch.manual_seed(2222)


