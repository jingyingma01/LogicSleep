import argparse
import random
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--cuda', default = 0, type = int, help = 'CUDA device number')
    parser.add_argument('--seed', type = int, default = 0, help = 'seed')
    parser.add_argument('--data', default = 'ISRUC_S3', help = 'data folder name')
    parser.add_argument('--num_node', type = int, default = 10, help = 'num of channels')
    parser.add_argument('--fold', type=int, default = 0, help='fold (0..10)')
    parser.add_argument('--batch', type=int, default = 8, help='batch size')
    parser.add_argument('--lr', type=float, default = 0.0005, help='learning rate')
    parser.add_argument('--lambda_crf', type=float, default=0.1, help='Weight λ for CRF loss term')
    parser.add_argument('--lambda_logic', type=float, default=0.01, help='Weight λ for Logic (DL2) loss term')

    parser.add_argument('--num_patch', type = int, default = 5, help='Number of Patch')
    parser.add_argument('--feat_dim', type = int, default = 600, help='Feature Dim')
    parser.add_argument('--norm', type = str, default = 'Batch', help='Batch/Layer/Group')
    parser.add_argument('--num_epochs', type = int, default = 60, help = 'epochs')
    parser.add_argument('--window', type = int, default = 16, help = 'window size')
    parser.add_argument('--overlap', type = int, default = 2, help = 'overlap size')
    parser.add_argument('--drop_n', type = float, default = 0.3, help = 'drop net')
    parser.add_argument('--drop_c', type = float, default = 0.5, help = 'drop output')
    parser.add_argument('--act_n', type = str, default = 'ELU', help = 'network act')
    parser.add_argument('--act_c', type = str, default = 'ELU', help = 'output act')
    parser.add_argument('--gcn_h', type = str, default = '1024 512 256 256', help = 'GCN hidden layer')
    parser.add_argument('--l_n', type = int, default = 3, help = 'The layer of Unet')
    parser.add_argument('--ks', type = str, default = '0.9 0.8 0.7')
    parser.add_argument('--cs', type = str, default = '0.5 0.5 0.5')
    parser.add_argument('--sch', type = int, default = 2, help = 'scheduler')
    parser.add_argument('--chs', type = str, default = '32 64 128 256')
    parser.add_argument('--kernal', type = str, default = '15 9 7 3', help = 'kernal')
    parser.add_argument('--delta_t', type = float, default = 0.8, help='Adjacency Time Matrix')
    parser.add_argument('--delta_p', type = float, default = 0.9, help='Adjacency Position Matrix')
    parser.add_argument('--num_class', type = int, default = 5, help = 'Number of Classification')
    parser.add_argument('--weightDecay', type = float, default = 0.005)
    parser.add_argument('--lrStepSize', type = int, default = 10)
    parser.add_argument('--lrGamma', type = float, default = 0.1)
    parser.add_argument('--lrFactor', type = float, default = 0.5)
    parser.add_argument('--lrPatience', type = int, default = 10)
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def app_run(args, config, G_data):
    net = GNet(config)
    trainer = Trainer(args, net, G_data)
    trainer.train()

def main():
    args = get_args()
    config = args
    print(config)
    set_random(config.seed)
    G_data = FileLoader(config).load_data(False)
    print('start training ------> fold', config.fold)
    app_run(args, config, G_data)

if __name__ == "__main__":
    main()