import argparse
from train import *

## Parser
parser = argparse.ArgumentParser(description='Training models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.002, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=1024, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default='./data', type=str, dest="data_dir")
parser.add_argument("--checkpoint", default='./result/svhn_1000/checkpoint1.pth.tar', type=str, dest="checkpoint")
parser.add_argument("--log_dir", default='./log/svhn_1000/', type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result/svhn_1000/", type=str, dest="result_dir")
parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')
parser.add_argument('--T', default=0.5, dest="T")
parser.add_argument('--n_labeled', default=1000, dest="n_labeled")
parser.add_argument('--lambda_u', default=250, dest="lambda_u")
parser.add_argument('--alpha', default=0.75, dest="alpha")
parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')
parser.add_argument('--ema_decay', default=0.999, dest="ema_decay")
parser.add_argument("--threshold", default=0.95, type=float, dest="threshold")

args = parser.parse_args()


if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
