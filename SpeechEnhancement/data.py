"""
Created on Sat Mar 5 2022
@author: Kuan-Lin Chen
"""
import glob
import math
import numpy as np
import torch
import librosa
import random
from torch.utils.data.dataset import Dataset
from datetime import datetime

MIXED_LENGTH = 64512 # the length used for training DNNs, a multiple of 1024, 4.032 seconds in 16000 Hz
N_FFT = 1024
FS = 16000
SIR_LU = [-10,20] # SIR lower and upper bounds
SNR_LU = [0,20] # SNR lower and upper bounds

def max_len(target,interference):
    return max(max([t.size(0) for t in target]),max([i.size(0) for i in interference]))

def mix(target,interference,sir,snr,l):
    """
    inputs:

    target: list of 1-d audio signals (torch tensors)
    interference: list of 1-d audio signals (empty list if no interference)
    sir: desired SIR
    snr: desired SNR (white Gaussian noise)
    l: the length after mixing

    outputs:
    y: normalized mixed signal (y=target signal + interference@SIR + white Gaussian noise@SNR) [1 x length]
    x: premixed signals [3 x length]    
    """
    # find the scaling factor (std) for the noise, assuming that all targets have unit stds
    std_n = math.sqrt(10**(-snr/10))
    # find the scaling factor (std) for all the interferences, the power is equally splitted into each interference
    std_i = math.sqrt(10**(-sir/10)/len(interference))
    # create a premixed placeholder x which is the decomposition of the output y, i.e., y = x[0,:] + x[1,:] + x[2,:] = target + interference + noise
    x = torch.zeros((3,l))
    # create a mixed placeholder y which is initially loaded with a noise
    x[2,:] = torch.randn((1,l))
    x[2,:] *= (std_n/torch.std(x[2,:]))
    # put the target and interference into the placeholder y
    for t in target:
        tmp = torch.zeros((1,l))
        s = random.randint(0,abs(l-t.size(0))) # random shift
        if l>=t.size(0):
            tmp[0,s:s+t.size(0)] = t
        else:
            tmp[0,:] = t[s:s+l]
        tmp /= torch.std(tmp) # normalization
        x[[0],:] += tmp
    for i in interference:
        tmp = torch.zeros((1,l))
        s = random.randint(0,abs(l-i.size(0))) # random shift
        if l>=i.size(0):
            tmp[0,s:s+i.size(0)] = i
        else:
            tmp[0,:] = i[s:s+l]
        tmp *= (std_i/torch.std(tmp)) # normalization
        x[[1],:] += tmp
    # mix the target, interference, and noise
    y = torch.sum(x,dim=0,keepdim=True)
    # normalization using the largest magnitude in the mixed signal
    max_abs = torch.abs(y).max()
    y /= max_abs
    x /= max_abs
    return y,x

class Magnitude2IRM(Dataset):
    def __init__(self,
                target_pattern,
                interference_pattern,
                num_target,
                num_interference,
                fs = FS,
                n_fft = N_FFT,
                sir_lu = SIR_LU,
                snr_lu = SNR_LU,
                ):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = self.n_fft // 2
        self.target_path_list = glob.glob(target_pattern)
        self.interference_path_list = glob.glob(interference_pattern)
        self.num_target_files = len(self.target_path_list)
        self.num_interference_files = len(self.interference_path_list)
        self.sir_lu = sir_lu
        self.snr_lu = snr_lu
        self.num_target = num_target
        self.num_interference = num_interference
        if self.num_target < 1 or self.num_interference < 1:
            raise AssertionError("num_target={}, num_interference={}, each of them should be larger than or equal to 1".format(self.num_target,self.num_interference))
        if self.num_target_files == 0 or self.num_interference_files == 0:
            raise FileNotFoundError("num_target_files={}, num_interference_files={}, pattern not found".format(self.num_target_files,self.num_interference_files))
        else:
            print("{} {} paths found in the pattern {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.num_target_files,target_pattern))
            print("{} {} paths found in the pattern {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.num_interference_files,interference_pattern))

        if self.num_target_files > self.num_interference_files:
            for k in range(self.num_target_files-self.num_interference_files):
                self.interference_path_list.append(self.interference_path_list[k%self.num_interference_files])
        else:
            for k in range(self.num_interference_files-self.num_target_files):
                self.target_path_list.append(self.target_path_list[k%self.num_target_files])
        self.num_pairs= max(self.num_target_files,self.num_interference_files)
        if self.num_target > self.num_pairs or self.num_interference > self.num_pairs:
            raise AssertionError("num_target={}, num_interference={}, each of them should be smaller than or equal to num_pairs {}".format(self.num_target,self.num_interference,self.num_pairs))
        # load all the data
        self.target_list = []
        self.interference_list = []
        print('{} Load the dataset into the memory... (this may take several minutes)'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),flush=True)
        for k in range(self.num_pairs):
            target_path = self.target_path_list[k]
            interference_path = self.interference_path_list[k]
            target = librosa.load(target_path,sr=self.fs,mono=True)[0]
            interference = librosa.load(interference_path,sr=self.fs,mono=True)[0]
            self.target_list.append(torch.from_numpy(target))
            self.interference_list.append(torch.from_numpy(interference))

        print('{} Number of examples: {}. Dataset is ready to be used'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.num_pairs),flush=True)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self,idx):
        target = self.target_list[idx]
        interference = self.interference_list[idx]
        n = random.randint(0,self.num_target-1)
        extra_target = random.sample(self.target_list[:idx]+self.target_list[idx+1:],n)
        extra_interference = random.sample(self.interference_list[:idx]+self.interference_list[idx+1:],n)
        # mix
        y,x = mix([target]+extra_target,[interference]+extra_interference,random.uniform(self.sir_lu[0],self.sir_lu[1]),random.uniform(self.snr_lu[0],self.snr_lu[1]),MIXED_LENGTH)
        # get the target signal x and the undesired signal z
        x = x[[0],:] # target signal
        z = y - x # undesired signal
        # STFT
        stft_y = torch.stft(input=y,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        stft_x = torch.stft(input=x,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        stft_z = torch.stft(input=z,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        # compute the magnitude of the input STFT
        abs_stft_y = torch.absolute(stft_y)
        # compute the ideal ratio mask
        pow_stft_x = torch.absolute(stft_x*torch.conj(stft_x))
        pow_stft_z = torch.absolute(stft_z*torch.conj(stft_z))
        irm = torch.sqrt(pow_stft_x/(pow_stft_x+pow_stft_z))
        # prevent 0/0 by replacing any nan with 0
        irm = torch.nan_to_num(irm)
        # data and label
        data = abs_stft_y.float()
        label = irm.float()
        return data,label

class Complex2IRM(Magnitude2IRM):
    def __getitem__(self,idx):
        target = self.target_list[idx]
        interference = self.interference_list[idx]
        # mix
        y,x = mix([target],[interference],random.uniform(self.sir_lu[0],self.sir_lu[1]),random.uniform(self.snr_lu[0],self.snr_lu[1]),MIXED_LENGTH)
        # get the target signal x and the undesired signal z
        x = x[[0],:] # target signal
        z = y - x # undesired signal
        # STFT
        stft_y = torch.stft(input=y,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=False)
        stft_x = torch.stft(input=x,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        stft_z = torch.stft(input=z,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        # compute the ideal ratio mask
        pow_stft_x = torch.absolute(stft_x*torch.conj(stft_x))
        pow_stft_z = torch.absolute(stft_z*torch.conj(stft_z))
        irm = torch.sqrt(pow_stft_x/(pow_stft_x+pow_stft_z))
        # prevent 0/0 by replacing any nan with 0
        irm = torch.nan_to_num(irm)
        # data and label
        data = torch.permute(stft_y.squeeze(0),(2,0,1))
        label = irm.float()
        return data,label

class SEDataset:
    def __init__(self,data,mode,num_target,num_interference):
        if "train" in mode:
            self.trainset = data(target_pattern = "./../../Datasets/timit_lowercase/train/*/*/*.wav",
                                 interference_pattern = "./../../Datasets/Nonspeech/train/*.wav",
                                 num_target = num_target,
                                 num_interference = num_interference)
        if "test" in mode:
            self.testset = data(target_pattern = "./../../Datasets/timit_lowercase/test/*/*/*.wav",
                                interference_pattern = "./../../Datasets/Nonspeech/test/*.wav",
                                num_target = num_target,
                                num_interference = num_interference)
        if "validation" in mode:
            self.validationset = data(target_pattern = "./../../Datasets/timit_lowercase/train/*/*/*.wav",
                                      interference_pattern = "./../../Datasets/Nonspeech/train/*.wav",
                                      num_target = num_target,
                                      num_interference = num_interference)
        if "debug" in mode:
            self.trainset = data(target_pattern = "./../../Datasets/timit_lowercase/train/dr1/*/*.wav",
                                 interference_pattern = "./../../Datasets/Nonspeech/train/*.wav",
                                 num_target = num_target,
                                 num_interference = num_interference)
            self.validationset = data(target_pattern = "./../../Datasets/timit_lowercase/train/dr1/*/*.wav",
                                      interference_pattern = "./../../Datasets/Nonspeech/train/*.wav",
                                      num_target = num_target,
                                      num_interference = num_interference)
        if not ("train" in mode or "validation" in mode or "test" in mode or "debug" in mode):
            raise AssertionError("incorrect mode, the mode should contain string train, validation, test, debug, or both like train_validation")

class Magnitude2IRMDataset(SEDataset):
    def __init__(self,mode="train_validation",num_target=1,num_interference=1):
        super().__init__(data=Magnitude2IRM,mode=mode,num_target=num_target,num_interference=num_interference)

class Complex2IRMDataset(SEDataset):
    def __init__(self,mode="train_validation",num_target=1,num_interference=1):
        super().__init__(data=Complex2IRM,mode=mode,num_target=num_target,num_interference=num_interference)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    se = Magnitude2IRMDataset(mode="debug",num_target=2,num_interference=2)
    dataloader = DataLoader(se.trainset,batch_size=8,shuffle=True,num_workers=1,pin_memory=False,drop_last=False)
    print(dataloader)
    print(len(dataloader))
    for idx, (data,label) in enumerate(dataloader):
        print(idx)
        print(data)
        print(type(data))
        print(data.shape)
        print(label)
        print(type(label))
        print(label.shape)
        break
    print('end of for loop')
    data = 20*np.log10(np.flip(np.squeeze(data[0,:,:,:].numpy(),axis=0),axis=0))
    print(data.shape)
    N_f_bin = data.shape[0]
    N_t_bin = data.shape[1]
    plt.imshow(data)
    plt.set_cmap('jet')
    plt.colorbar(label="spectrogram")
    plt.clim(-60,0)
    plt.show()
