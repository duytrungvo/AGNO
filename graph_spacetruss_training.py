import torch
import argparse
import yaml

from lib.model_graph import GNN, AGNN, GNO, AGNO, New_model
from lib.utils_spacetruss_train import sup_graph_train, sup_graph_attn_train
from lib.utils_data import generate_graphgridspacetruss_data_loader
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
    parser.add_argument('--model', type=str, default='graph')
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
    if args.model == 'GNN':
        model = GNN(config)
    if args.model == 'AGNN':
        model = AGNN(config)
    if args.model == 'GNO':
        model = GNO(config)
    if args.model == 'AGNO':
        model = AGNO(config)
    if args.model == 'self_defined':
        model = New_model(config)

    # load the data
    train_loader, val_loader, test_loader = generate_graphgridspacetruss_data_loader(args, config)

    # then train solution function
    if args.model == 'AGNO' or args.model == 'AGNN':
        sup_graph_attn_train(args, config, model, device, (train_loader, val_loader, test_loader))
    else:
        sup_graph_train(args, config, model, device, (train_loader, val_loader, test_loader))
