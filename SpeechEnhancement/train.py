"""
Created on Fri Mar 5 2021

@author: Kuan-Lin Chen
"""
import sys
import os
import time
import torch
import scipy.io
import math
from datetime import datetime
from utils import get_device_name

class TrainParam:
    def __init__(self,
                mu,
                mu_scale,
                mu_epoch,
                weight_decay,
                momentum,
                batch_size,
                nesterov
                ):
        assert len(mu_scale)==len(mu_epoch), "the length of mu_scale and mu_epoch should be the same"        
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epoch = mu_epoch[-1]
        self.mu = mu
        self.mu_scale = mu_scale
        self.mu_epoch = mu_epoch
        self.nesterov = nesterov
                 
class TrainRegressor:
    num_workers = 4
    pin_memory = False
    ckpt_filename = 'train.pt'
    def __init__(self,
                name,
                net,
                tp,
                trainset,
                validationset,
                criterion,
                device,
                seed,
                resume,
                checkpoint_folder,
                milestone = [],
                print_every_n_batch = 1
                ):
        torch.manual_seed(seed)
        self.criterion = criterion #torch.nn.L1Loss(reduction='none') # MSELoss or L1Loss
        net = net()
        self.checkpoint_folder = checkpoint_folder
        self.name = name
        self.seed = seed
        self.milestone = milestone
        self.print_every_n_batch = print_every_n_batch
        self.device = device
        self.net = net.to(device)
        print("{} {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),get_device_name(device)))
        self.num_parameters = self.count_parameters()
        print("Number of parameters for the model {}: {:,}".format(name,self.num_parameters))

        self.mu_lambda = lambda i: next(tp.mu_scale[j] for j in range(len(tp.mu_epoch)) if min(tp.mu_epoch[j]//(i+1),1.0) >= 1.0) if i<tp.max_epoch else 0
        self.rho_lambda = lambda i: next(tp.rho_scale[j] for j in range(len(tp.rho_epoch)) if min(tp.rho_epoch[j]//i,1.0) >= 1.0) if i>0 else 0

        self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=tp.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        self.validationloader = torch.utils.data.DataLoader(validationset,batch_size=tp.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        self.optimizer = torch.optim.SGD(self.net.parameters(),lr=tp.mu,momentum=tp.momentum,nesterov=tp.nesterov,weight_decay=tp.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.mu_lambda)
        self.tp = tp
        self.total_train_time = 0
        self.start_epoch = 1
        self.train_loss = []
        self.validation_loss = []
        self.best_validation_loss = sys.float_info.max
        self.ckpt_path = self.checkpoint_folder+self.name+'/'+self.ckpt_filename

        if resume is True and os.path.isfile(self.ckpt_path):
            print('Resuming {} from a checkpoint at {}'.format(self.name,self.ckpt_path),flush=True)
            self.__load()
        else:
            print('Ready to train {} from scratch...'.format(self.name),flush=True)
            init_validation_loss = self.validation()
            self.init_validation_loss = init_validation_loss
            self.best_validation_loss = init_validation_loss
            self.__save_net('init_model.pt')
            self.__save(0)
 
    def __get_lr(self):
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def __check_folder(self):
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(self.checkpoint_folder+self.name):
            os.mkdir(self.checkpoint_folder+self.name)

    def __load(self):
        # Load checkpoint.
        checkpoint = torch.load(self.ckpt_path,map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_validation_loss = checkpoint['best_validation_loss']
        self.start_epoch = checkpoint['epoch']+1
        self.train_loss = checkpoint['train_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.total_train_time = checkpoint['total_train_time']
        self.init_validation_loss = checkpoint['init_validation_loss']

    def __save_net(self,filename):
        self.__check_folder()
        net_path = self.checkpoint_folder+self.name+'/'+filename
        torch.save(self.net.state_dict(), net_path)
        print('{} model saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),net_path))

    def __save(self,epoch):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'init_validation_loss': self.init_validation_loss,
            'best_validation_loss': self.best_validation_loss,
            'epoch': epoch,
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'num_param': self.num_parameters,
            'seed': self.seed,
            'mu': self.tp.mu,
            'mu_scale': self.tp.mu_scale,
            'mu_epoch': self.tp.mu_epoch,
            'weight_decay': self.tp.weight_decay,
            'momentum': self.tp.momentum,
            'batch_size': self.tp.batch_size,
            'total_train_time': self.total_train_time,
            }
        self.__check_folder()
        torch.save(state, self.ckpt_path)
        print('{} checkpoint saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.ckpt_path))
        del state['net'], state['optimizer'], state['scheduler']
        state_path = self.checkpoint_folder+self.name+'/train.mat'
        scipy.io.savemat(state_path,state)
        print('{} state saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),state_path))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train(self):
        for i in range(self.start_epoch,self.tp.max_epoch+1):
            lr = self.__get_lr()
            num_batch = len(self.trainloader)
            tic = time.time()
            train_loss = self.__train(i)
            toc = time.time()
            self.total_train_time += (toc-tic)
            print('training speed: {:.3f} seconds/epoch'.format(self.total_train_time/i))
            validation_loss = self.validation()
            print('{} [{}] [Validation] epoch: {}/{} batch: {:3d}/{} lr: {:.1e} loss: {:.4f} best: {:.4f}'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.name,
                    i,
                    self.tp.max_epoch,
                    num_batch,
                    num_batch,
                    lr,
                    validation_loss,
                    min(self.best_validation_loss,validation_loss),
                    self.total_train_time
                    ),flush=True)
 
            self.train_loss.append(train_loss)
            self.validation_loss.append(validation_loss)
            
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.__save_net('best_model.pt')

            for k in self.milestone:
                if k==i:
                    self.__save_net('epoch_'+str(k)+'_model.pt')
                    self.__save(k)

            if math.isnan(train_loss):
                print("{} NaN train loss... break the training loop".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                break
        if self.start_epoch<self.tp.max_epoch+1:
            self.__save_net('last_model.pt')
            self.__save(i)
            print('end of training at epoch {} for the model saved at {}'.format(i,self.ckpt_path))
        else:
            print('the model '+self.ckpt_path+' has already been trained for '+str(self.tp.max_epoch)+' epochs')
        return self
    
    def __train(self,epoch_idx):
        tic = time.time()
        self.net.train()
        accumulated_train_loss = 0
        total = 0
        torch.manual_seed(self.seed+epoch_idx)
        lr = self.__get_lr()
        num_batch = len(self.trainloader)
        for batch_idx, (inputs, targets) in enumerate(self.trainloader,1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            batch_mean_loss = torch.mean(loss)

            if torch.isnan(batch_mean_loss):
                print('Nan train loss detected. The previous train loss: {:.4f}'.format(train_loss))
                return float("nan")

            batch_mean_loss.backward()
            self.optimizer.step()

            accumulated_train_loss += torch.sum(loss).item()

            total += loss.numel()

            train_loss = accumulated_train_loss/total
            toc = time.time()
            if (batch_idx-1)%self.print_every_n_batch == 0 or batch_idx == num_batch:
                print('{} [{}] [Train] epoch: {}/{} batch: {:3d}/{} lr: {:.1e} loss: {:.4f} | ELA: {:.1f}s'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.name,
                    epoch_idx,
                    self.tp.max_epoch,
                    batch_idx,
                    num_batch,
                    lr,
                    train_loss,
                    self.total_train_time+toc-tic
                    ),flush=True)
        self.scheduler.step()
        return train_loss
    
    def validation(self):
        self.net.eval()
        accumulated_validation_loss = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.validationloader,1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                
                accumulated_validation_loss += torch.sum(loss).item()
                total += loss.numel()
                
        validation_loss = accumulated_validation_loss/total
        return validation_loss
    
if __name__ == '__main__':
    from data import Complex2IRMDataset
    from models import *
    from test import *
   
    # setting
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0)) # print the GPU
    se = Complex2IRMDataset()
    seed = 0
    device = "cuda:0"

    tp = TrainParam(
        mu=0.05,
        mu_scale=[1.0,0.2],#[1.0,0.2,0.04,0.008],
        mu_epoch=[2,4],#[60,120,160,200],
        weight_decay=5e-4,
        momentum=0.9,
        batch_size = 4,
        nesterov = True
        )
    
    # initiate a model
    c = TrainRegressor(
        name='COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny_seed='+str(seed),
        net=COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny,
        tp=tp,
        trainset=se.trainset,
        validationset=se.trainset,
        device=device,
        seed=seed,
        resume=False,
        checkpoint_folder = './checkpoint-dryrun/',
        milestone = [40,100,160],
        print_every_n_batch = 1
        )
 
    # train
    c.train()

    # load the model after training
    net = COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny()
    pretrained_model = torch.load(c.checkpoint_folder+c.name+'/last_model.pt',map_location=device)
    net.load_state_dict(pretrained_model,strict=True)
    net = net.to(device) 

    # test
    testRegressor(net,se.testset,device,c.checkpoint_folder+c.name,1,1)
