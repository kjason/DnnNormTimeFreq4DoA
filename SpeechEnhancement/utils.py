"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import os
import torch

def dir_path(path):
    if path[-1]=='/':
        if not os.path.isdir(path):
            os.mkdir(path)
        return path
    else:
        raise NotADirectoryError(path)

def check_device(device):
    if device == 'cpu':
        return device
    elif torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            if device == 'cuda:'+str(i):
                return device
        raise ValueError('{} not found in the available cuda or cpu list'.format(device))
    else:
        raise ValueError('{} is not a valid cuda or cpu device'.format(device))

def get_device_name(device):
    if device[:4] == 'cuda':
        return torch.cuda.get_device_name(int(device[-1])) # print the GPU
    else:
        return device

