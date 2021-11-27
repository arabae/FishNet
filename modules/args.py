import argparse
import numpy as np

parser = argparse.ArgumentParser()    # make parser

# get arguments
def get_args():
    args, unparsed = parser.parse_known_args()
    return args

# return bool type of argument
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# Data Preprocess Arguments
parser.add_argument("--data_batch", type=int, default=5, help="number of divided training batches")
parser.add_argument("--num_data_batch", type=int, default=10000, help="number of data of divided training batches")
parser.add_argument("--data_path", type=str, default="./cifar-10-batches-py", help="dataset directory")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes")

# Model Parameters
parser.add_argument("--batch_size", type=int, default=64, help="number of batch")
parser.add_argument("--n_tail", type=int, default=3, help="number of tail modules")
parser.add_argument("--n_body", type=int, default=3, help="number of body modules")
parser.add_argument("--n_head", type=int, default=3, help="number of head modules")
parser.add_argument("--in_channel", type=int, default=64, help="dimension of channels")
parser.add_argument("--model_path", type=str, default="./results", help="model directory to save or load")

# Training Parameters
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument("--isTrain", type=str2bool, default=True, help="choose Train(true) or Not(false)")
parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--epoch", type=int, default=100, help="number of max epochs")

# wandb
parser.add_argument("--wandb_project_name", type=str, default='FishNet', help="wandb project name")
parser.add_argument("--wandb_run_name", type=str, default='baseline', help="wandb run name")
