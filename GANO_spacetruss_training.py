import torch
import argparse
import yaml

from lib.model_spacetruss import GANO, New_model_truss
from lib.utils_spacetruss_train import sup_train
from lib.utils_data import generate_spacetruss_data_loader_var_load
import random
import numpy as np
import os

def seed_everything(seed: int = 2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For cuBLAS GEMM determinism on CUDA (PyTorch docs):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may error if a non-deterministic op is used):
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Turn off autotuner & TF32 so kernels don’t change
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Optional: fewer threads to avoid tiny scheduling diffs
    # torch.set_num_threads(1)

if __name__ == "__main__":

    seed_everything(2025)
    # define arguements
    parser = argparse.ArgumentParser(description='command setting')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--data', type=str, default='spacetruss')
    parser.add_argument('--model', type=str, default='GANO')
    parser.add_argument('--geo_node', type=str, default='vary_bound_sup', choices=['vary_bound_sup', 'all_domain'])
    args = parser.parse_args()
    print('Model forward phase: {}'.format(args.phase))
    print('Using dataset: {}'.format(args.data))
    print('Using model: {}'.format(args.model))

    # extract configuration
    with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    print('Data name: {}'.format(config['data']['datapath']))

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    if args.model == 'GANO':
        model = GANO(config)
    if args.model == 'self_defined':
        model = New_model_truss(config)

    # load the data
    train_loader, val_loader, test_loader, num_nodes_list = generate_spacetruss_data_loader_var_load(args, config)

    # then train solution function
    sup_train(args, config, model, device, (train_loader, val_loader, test_loader), num_nodes_list)
