"""
Created on Sat Mar 5 2022
@author: Kuan-Lin Chen
"""
import torch
from SpeechEnhancement.models import model_dict 
from SpeechEnhancement.data import mix,MIXED_LENGTH

class Predictor():
    def __init__(self,name,model_path,fs=16000,n_fft=1024,device='cuda:0'):
        self.name = name
        self.model_path = model_path
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = self.n_fft // 2
        self.device = device

        # load the model
        pretrained_model = torch.load(self.model_path,map_location=device)
        self.net = model_dict[self.name]()
        self.net.load_state_dict(pretrained_model,strict=True)
        self.net = self.net.to(self.device)

    def _stft_predict_irm(self,stft_y):
        """
        input: stft_y is a complex torch tensor on the device with size [C x F x T] or [F x T] where C is the number of channels, F is the number of frequency bins, and T is the number of frames
        output: the estimated ideal ratio mask (IRM) using the specified pretrained DNN, a real torch tensor of size [C x F x T] or [F x T] on the device
        """
        with torch.no_grad():
            if "MAG" in self.name:
                if len(stft_y.size()) == 3:
                    stft_y = stft_y.unsqueeze(1)
                elif len(stft_y.size()) == 2:
                    stft_y = stft_y.unsqueeze(0).unsqueeze(0)
                else:
                    raise AssertionError("invalid dimension in the MAG mode")
                y = torch.absolute(stft_y).float()
            elif "COMPLEX" in self.name:
                if len(stft_y.size()) == 2:
                    stft_y = stft_y.unsqueeze(0)
                y = torch.stack([stft_y.real,stft_y.imag],dim=1).float()
            else:
                raise AssertionError("invalid model name, must contain MAG or COMPLEX")
            irm = self.net(y)
        return irm.squeeze()

    def _get_irm_stft(self,y):
        """
        input: y is a real numpy array on the CPU with size [C x L] or [L] where C is the number of channels and L is the length of the signal in the time domain
        output: IRM and the STFT of the signal y, both on the device, irm is a real torch tensor of size [C x F x T] or [F x T] and stft_y is a complex torch tensor of size [C x F x T] or [F x T]
        """
        y = torch.from_numpy(y).to(self.device)
        stft_y = torch.stft(input=y,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft).to(self.device),onesided=True,return_complex=True)
        irm = self._stft_predict_irm(stft_y)
        return irm,stft_y

    def irm(self,y):
        """
        input: see the input in the member function _get_irm_stft
        output: IRM, a real numpy array of size [C x F x T] or [F x T] on the CPU
        """
        return self._get_irm_stft(y)[0].cpu().numpy()

    def stft(self,y):
        """
        input: see the input in the member function _get_irm_stft
        output: STFT of the signal y on the CPU, a complex numpy array of size [C x F x T] or [F x T]
        """
        return self._get_irm_stft(y)[1].cpu().numpy()

    def enhance(self,y):
        """
        input: see the input in the member function _get_irm_stft
        output: the enhanced signal in the time-domain, a real numpy array of size [C x L] or [L]
        """
        irm, stft_y = self._get_irm_stft(y)
        enhanced_stft = irm*stft_y
        enhanced_time = torch.istft(input=enhanced_stft,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft).to(self.device),onesided=True)
        return enhanced_time.cpu().numpy()

    def get_gt(self,target,interference,sir,snr):
        """
        input: target is a real numpy array of size [L_1], interference is a real numpy array of size [L_2], sir and snr are real numbers
        output: IRM of size [F x T], the enhanced [L], mixed [L], and target [L] signals in the time domain, they are all real numpy array on the CPU
        """
        target = torch.from_numpy(target)
        interference = torch.from_numpy(interference)
        y,x = mix([target],[interference],sir,snr,MIXED_LENGTH)
        x = x[[0],:]
        z = y - x
        # the DNN will take y and reconstruct x
        stft_y = torch.stft(input=y,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        stft_x = torch.stft(input=x,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        stft_z = torch.stft(input=z,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True,return_complex=True)
        # compute the ideal ratio mask
        pow_stft_x = torch.absolute(stft_x*torch.conj(stft_x))
        pow_stft_z = torch.absolute(stft_z*torch.conj(stft_z))
        irm = torch.sqrt(pow_stft_x/(pow_stft_x+pow_stft_z))
        enhanced_stft = irm*stft_y
        enhanced = torch.istft(input=enhanced_stft,n_fft=self.n_fft,hop_length=self.hop_length,window=torch.hann_window(self.n_fft),onesided=True)
        enhanced = enhanced.float().squeeze().cpu().numpy()
        irm = irm.float().squeeze().cpu().numpy()
        y = y.float().squeeze().cpu().numpy()
        target = x.squeeze().cpu().numpy()
        return irm, enhanced, y, target 

if __name__ == '__main__':
    import librosa
    import random
    import numpy as np
    from scipy.io import savemat

    random.seed(0)
    torch.manual_seed(0)
    # hyperparameters
    fs = 16000
    snr = 20
    sir = 5
    # specify the clean speech and the interference
    #clean_path = './../Datasets/timit_lowercase/train/dr3/mbef0/si1911.wav'
    #inter_path = './../Datasets/Nonspeech/train/n75.wav'
    #clean_path = './../Datasets/timit_lowercase/train/dr1/fcjf0/sa2.wav'
    #inter_path = './../Datasets/Nonspeech/train/n60.wav'
    clean_path1 = './../Datasets/timit_lowercase/test/dr1/faks0/si1573.wav'
    inter_path1 = './../Datasets/Nonspeech/test/n33.wav'

    clean_path2 = '../Datasets/timit_lowercase/test/dr2/fcmr0/si1105.wav'
    inter_path2 = './../Datasets/Nonspeech/test/n70.wav'

    # read clean and interference audio files
    #clean = librosa.load(clean_path,sr=fs,mono=True)[0]
    #inter = librosa.load(inter_path,sr=fs,mono=True)[0]

    clean1 = librosa.load(clean_path1,sr=fs,mono=True)[0]
    inter1 = librosa.load(inter_path1,sr=fs,mono=True)[0]

    clean2 = librosa.load(clean_path2,sr=fs,mono=True)[0]
    inter2 = librosa.load(inter_path2,sr=fs,mono=True)[0]

    clean = np.zeros(max(clean1.size,clean2.size))
    inter = np.zeros(max(inter1.size,inter2.size))
    clean[:clean1.size] = clean1
    clean[:clean2.size] += clean2
    inter[:inter1.size] = inter1
    inter[:inter2.size] += inter2

    # specify the pretrained DNN model
    name = 'COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny'
    folder_name = 'COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny_loss=L1_mu=0.1_bs=16_nt=3_ni=3_seed=0'
    predictor = Predictor(name=name,model_path='./SpeechEnhancement/checkpoint/'+folder_name+'/last_model.pt')

    # save the results to the following path
    result_path = 'results_net={}_snr={}_sir={}.mat'.format(folder_name,snr,sir)

    # get the ground truth IRM, the enhanced signal using the ground truth IRM, mixed signal, and the target signal
    gt_irm, gt_enhanced, y, target = predictor.get_gt(target=clean,interference=inter,sir=sir,snr=snr)
    # get the estimated IRM and the enhanced signal
    irm = predictor.irm(y)
    enhanced = predictor.enhance(y)
    
    # save the results
    savemat(result_path,{'fs':fs,'snr':snr,'sir':sir,'irm':irm,'gt_irm':gt_irm,'target':target,'enhanced':enhanced,'gt_enhanced':gt_enhanced,'y':y})

    print('[Setting] SNR: {}, SIR: {}, Network: {}\nResults saved at ./{}'.format(snr,sir,predictor.name,result_path))
