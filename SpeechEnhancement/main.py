"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import argparse
from datetime import datetime
from data import Magnitude2IRMDataset,Complex2IRMDataset
from models import model_dict
from train import TrainParam,TrainRegressor
from utils import dir_path,check_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DNN model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='COMPLEX_IRM', choices=['MAG_IRM','COMPLEX_IRM'], help='speech enhancement datasets')
    parser_resume_group = parser.add_mutually_exclusive_group()
    parser_resume_group.add_argument('--resume', dest='resume', action='store_true', help='resume from the last checkpoint',default=True)
    parser_resume_group.add_argument('--no-resume', dest='noresume', action='store_true', help='start a new training or overwrite the last one',default=False)
    parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
    parser.add_argument('--device', default='cuda:0', type=check_device, help='specify a CUDA or CPU device, e.g., cuda:0, cuda:1, and cpu')
    parser.add_argument('--mu', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser_nesterov_group = parser.add_mutually_exclusive_group()
    parser_nesterov_group.add_argument('--nesterov', dest='nesterov', action='store_true', help='enable Nesterov momentum',default=True)
    parser_nesterov_group.add_argument('--no-nesterov', dest='nonesterov', action='store_true', help='disable Nesterov momentum',default=False)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--mu_scale', default=[1.0,0.2,0.04,0.008], nargs='+', type=float, help='learning rate scaling')
    parser.add_argument('--mu_epoch', default=[60,120,160,200], nargs='+', type=int, help='learning rate scheduling')
    parser.add_argument('--milestone', default=[40,100,150], nargs='+', type=int, help='the model trained after these epochs will be saved')
    parser.add_argument('--print_every_n_batch', default=10, type=int, help='print the training status every n batch')
    parser.add_argument('--seed_list', default=[0], nargs='+', type=int, help='train a model with different random seeds')
    parser.add_argument('--model', default='COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny', choices=list(model_dict.keys()), help='the DNN model')
    parser.add_argument('--n_target', default=3, type=int, help='number of targets in the mixed signal')
    parser.add_argument('--n_interference', default=3, type=int, help='number of interferences in the mixed signal')
    parser.add_argument('--loss', default='L1', choices=['L1','MSE'], help='loss function')
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True

    print('{} [main.py]'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    if args.data == 'MAG_IRM':
        dataset = Magnitude2IRMDataset(num_target=args.n_target,num_interference=args.n_interference)
    elif args.data == 'COMPLEX_IRM':
        dataset = Complex2IRMDataset(num_target=args.n_target,num_interference=args.n_interference)
    else:
        raise AssertionError('invalid dataset')

    trainset = dataset.trainset
    validationset = dataset.validationset

    if args.loss == 'L1':
        criterion = torch.nn.L1Loss(reduction='none')
    elif args.loss == 'MSE':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise AssertionError('invalid loss function')

    for seed in args.seed_list:
        name = "{}_loss={}_mu={}_bs={}_nt={}_ni={}_seed={}".format(args.model,args.loss,args.mu,args.batch_size,args.n_target,args.n_interference,seed)
        tp = TrainParam(
            mu=args.mu,
            mu_scale=args.mu_scale,
            mu_epoch=args.mu_epoch,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            batch_size = args.batch_size,
            nesterov = args.nesterov and not args.nonesterov
            )
        c = TrainRegressor(
            name=name,
            net=model_dict[args.model],
            tp=tp,
            trainset=trainset,
            validationset=validationset,
            criterion = criterion,
            device=args.device,
            seed=seed,
            resume=args.resume and not args.noresume,
            checkpoint_folder = args.checkpoint_folder,
            milestone = args.milestone,
            print_every_n_batch = args.print_every_n_batch
        ).train()
        print('{} [main.py] training for the model {} is completed'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),name))
