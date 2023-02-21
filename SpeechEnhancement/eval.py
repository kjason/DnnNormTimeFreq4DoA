"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import argparse
from data import Magnitude2IRMDataset,Complex2IRMDataset
from models import model_dict
from test import testRegressor
from utils import dir_path,check_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the performance on a given DNN model and report the test loss',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='COMPLEX_IRM', choices=['MAG_IRM','COMPLEX_IRM'], help='speech enhancement datasets')
    parser.add_argument('--mode', default='last', choices=['last','best','init'], help='evaluation mode (last: the last epoch, best: best performance on the validation set, init: after initialization)')
    parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
    parser.add_argument('--device', default='cuda:0', type=check_device, help='specify a CUDA or CPU device, e.g., cuda:0, cuda:1, and cpu')
    parser.add_argument('--mu', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size used in the pretrained model')
    parser.add_argument('--eval_batch_size', default=16, type=int, help='batch size for evaluation (a smaller evaluation batch size may be required for a GPU with less memory)')
    parser.add_argument('--seed_list', default=[0], nargs='+', type=int, help='train a model with different random seeds')
    parser.add_argument('--model', default='COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny', choices=list(model_dict.keys()), help='the DNN model')
    parser.add_argument('--net_n_target', default=3, type=int, help='number of targets in the mixed signal in the pre-trained model')
    parser.add_argument('--net_n_interference', default=3, type=int, help='number of interferences in the mixed signal in the pre-trained model')
    parser.add_argument('--n_target', default=1, type=int, help='number of targets in the mixed signal used to test')
    parser.add_argument('--n_interference', default=1, type=int, help='number of interferences in the mixed signal used to test')
    parser.add_argument('--loss', default='L1', choices=['L1','MSE'], help='loss function')
    parser.add_argument('--net_loss', default='L1', choices=['L1','MSE'], help='the loss function used in the pretrained model')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True

    if args.data == 'MAG_IRM':
        testset = Magnitude2IRMDataset(mode="test",num_target=args.n_target,num_interference=args.n_interference).testset
    elif args.data == 'COMPLEX_IRM':
        testset = Complex2IRMDataset(mode="test",num_target=args.n_target,num_interference=args.n_interference).testset
    else:
        raise AssertionError('invalid dataset')

    if args.loss == 'L1':
        criterion = torch.nn.L1Loss(reduction='none')
    elif args.loss == 'MSE':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise AssertionError('invalid loss function')

    for seed in args.seed_list:
        name = "{}_loss={}_mu={}_bs={}_nt={}_ni={}_seed={}".format(args.model,args.net_loss,args.mu,args.batch_size,args.net_n_target,args.net_n_interference,seed)
        print('start testing the model '+name)
        net = model_dict[args.model]()
        pretrained_model = torch.load(args.checkpoint_folder+name+'/'+args.mode+'_model.pt',map_location=args.device)
        net.load_state_dict(pretrained_model,strict=True)
        testRegressor(net,testset,criterion,args.device,args.checkpoint_folder+name,args.n_target,args.n_interference,args.eval_batch_size)
