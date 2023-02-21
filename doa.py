"""
Created on Sat Mar 5 2022
@author: Kuan-Lin Chen
"""
import os
import glob
import torch
import time
import scipy.io
import datetime
import argparse
import numpy as np
from pyroomacoustics import doa
from weighted import Weighted
from principal import Principal
from predict import Predictor
from chamferdist import ChamferDistance

def dir_path(path):
    if path[-1]=='/':
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

def getDoAResults(
                rt60,
                snr,
                sir,
                n_src,
                n_tgt,
                model_name,
                model_path,
                device,
                result_prefix,
                n_snapshots,
                deg_step,
                enabled_algo,
                irm_thres,
                dataset_prefix,
                num_files,
                save_spectra,
                freq_range,
                acc_deg_thres
                ):

    # the dataset setting
    setting_path = "{}setting.npy".format(dataset_prefix)
    # all the data samples in the dataset that match the given condition such as the number of sources and targets, rt60, SNR, and SIR
    pattern = '{}nsrc={}_ntgt={}_rt60={}_snr={}_sir={}/*.npy'.format(dataset_prefix,n_src,n_tgt,rt60,snr,sir)
    # the results will be saved to this path
    result_path = '{}model={}_nsrc={}_ntgt={}_rt60={}_snr={}_sir={}_nsnap={}_thres={}_freq_range={}_{}_accthres={}.mat'.format(result_prefix,model_name,n_src,n_tgt,rt60,snr,sir,n_snapshots,irm_thres,freq_range[0],freq_range[1],acc_deg_thres)

    # skip if the result path already exists
    if os.path.exists(result_path):
        print("the result path {} already exists, skip this evaluation".format(result_path),flush=True)
        return None
    else:
        print("the result doesn't exist, start evaluating the data with pattern {}".format(pattern),flush=True)

    # load the setting of the dataset
    if not os.path.exists(setting_path):
        raise FileNotFoundError("not found the setting at "+setting_path)

    setting = np.load(setting_path,allow_pickle=True).item()
    path_list = glob.glob(pattern)

    if len(path_list) == 0:
        raise FileNotFoundError("the pattern could not find any data")

    # we set the number of files to be the given number to perform C trials
    if num_files == '*':
        num_files = len(path_list)
    else:
        if int(num_files) > len(path_list):
            raise AssertionError("num_files is larger than the number of files in the dataset")
        num_files = int(num_files)

    # location of the microphone array
    mic_locs = np.c_[tuple(setting['mic_locs'])]
    # sampling frequency
    fs = setting['fs']
    # a placeholder for the received signal at the microphone array
    data = []

    # load C files
    for path in path_list[:num_files]:
        obj = np.load(path,allow_pickle=True).item()
        # sph is the spherical coordinate, i.e., the length r, the polar angle theta, the azimuthal angle phi
        # mix is the received signal at the microphone array
        data.append([obj['sph'],obj['mix']])

    # create the result folder if not already exists
    if not os.path.exists(result_prefix):
        os.mkdir(result_prefix)

    # the pre-trained signal enhancement model (a DNN)
    p = Predictor(name=model_name,model_path=model_path,device=device)
    # The received signal will be truncated to a length equivalent to the given number of snapshots
    snap_length = int(n_snapshots*(p.n_fft//2-1))
    # hyperparameters for the DoA estimation algorithms below
    kwargs = {
        'L': mic_locs,
        'fs': fs, 
        'nfft': p.n_fft,
        'c': 343,
        'mode': "far",
        'azimuth': np.deg2rad(np.arange(360,step=deg_step))}

    # DoA estimation algorithms
    algorithms = {
        'MUSIC': doa.music.MUSIC(**kwargs), # MUSIC
        'DNN_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNmin_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNmax_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNmean_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNmedian_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNprod_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNgeom_MUSIC': doa.music.MUSIC(**kwargs),
        'DNNthres_MUSIC': doa.music.MUSIC(**kwargs),

        'Principal': Principal(**kwargs), # the principal vector method
        'DNN_Principal': Principal(**kwargs),
        'DNNmin_Principal': Principal(**kwargs),
        'DNNmax_Principal': Principal(**kwargs),
        'DNNmean_Principal': Principal(**kwargs),
        'DNNmedian_Principal': Principal(**kwargs),
        'DNNprod_Principal': Principal(**kwargs),
        'DNNgeom_Principal': Principal(**kwargs),
        'DNNthres_Principal': Principal(**kwargs),

        'Weighted': Weighted(**kwargs), # the SRP method
        'DNN_Weighted': Weighted(**kwargs),
        'DNNmin_Weighted': Weighted(**kwargs),
        'DNNmax_Weighted': Weighted(**kwargs),
        'DNNmean_Weighted': Weighted(**kwargs),
        'DNNmedian_Weighted': Weighted(**kwargs),
        'DNNprod_Weighted': Weighted(**kwargs),
        'DNNgeom_Weighted': Weighted(**kwargs),
        'DNNthres_Weighted': Weighted(**kwargs),

        'EngWeighted': Weighted(**kwargs), # the normalized T-F weighted method
        'DNN_EngWeighted': Weighted(**kwargs),
        'DNNmin_EngWeighted': Weighted(**kwargs),
        'DNNmax_EngWeighted': Weighted(**kwargs),
        'DNNmean_EngWeighted': Weighted(**kwargs),
        'DNNmedian_EngWeighted': Weighted(**kwargs),
        'DNNprod_EngWeighted': Weighted(**kwargs),
        'DNNgeom_EngWeighted': Weighted(**kwargs),
        'DNNthres_EngWeighted': Weighted(**kwargs)
        }

    # we will only run DoA estimation algorithms that are in the enabled_algo list
    if 'All' not in enabled_algo:
        for key in list(algorithms.keys()):
            if key not in enabled_algo:
                algorithms.pop(key)

    # placeholders
    predictions = {n:[] for n in list(algorithms.keys())}
    predictions_np = {n:[] for n in list(algorithms.keys())}
    gt = []
    interference_doa = []
    spatial_spectra = {n:[] for n in list(algorithms.keys())}


    # start running the signal enhancement model and DoA estimation algorithms

    # tic
    t0 = time.time()

    for sph, mix in data:
        gt.append(torch.Tensor([sph[i][2] for i in range(n_tgt)]))
        interference_doa.append(torch.Tensor([sph[i][2] for i in range(n_tgt,n_src)]).rad2deg().cpu().numpy())
        irm,stft_signals = p._get_irm_stft(mix[:,:snap_length])
        for algo_name, algo in algorithms.items():
            num_tgt = n_tgt # assume that the number of targets is known
            if 'Eng' in algo_name:
                eng_norm = torch.linalg.vector_norm(stft_signals,dim=0)
                stft_signals = stft_signals/eng_norm
            if 'DNN' in algo_name:
                if 'min' in algo_name:
                    m_irm = torch.min(irm,dim=0)[0]
                elif 'max' in algo_name:
                    m_irm = torch.max(irm,dim=0)[0]
                elif 'mean' in algo_name:
                    m_irm = torch.mean(irm,dim=0)
                elif 'median' in algo_name:
                    m_irm = torch.from_numpy(np.median(irm.cpu().numpy(),axis=0)).to(device)
                elif 'prod' in algo_name:
                    m_irm = torch.prod(irm,dim=0)
                elif 'geom' in algo_name:
                    m_irm = torch.pow(torch.prod(irm,dim=0),1/irm.shape[0])
                elif 'thres' in algo_name:
                    m_irm = (irm>irm_thres).float()
                else:
                    m_irm = irm
                snapshot = m_irm*stft_signals
            else:
                snapshot = stft_signals
            snapshot = snapshot.cpu().numpy()
            algo.locate_sources(snapshot,num_src=num_tgt,freq_range=freq_range)
            azi = algo.azimuth_recon.tolist()
            if len(azi) == 0:
                raise AssertionError("no peaks found")
            azi = azi + azi*(n_tgt - len(azi))
            predictions[algo_name].append(torch.Tensor(azi).float())
            if save_spectra is True:
                spatial_spectra[algo_name].append(algo.Pssl)

    # toc
    t1 = time.time()
    elapsed = t1 - t0

    # process the results
    gt = torch.stack(gt,dim=0).rad2deg().unsqueeze(2)
    for algo_name in algorithms.keys():
        predictions[algo_name] = torch.stack(predictions[algo_name],dim=0).rad2deg().unsqueeze(2)
        predictions_np[algo_name] = predictions[algo_name].cpu().numpy()

    # compute the accuracy
    ACC = {}
    acc_gt = gt.squeeze(-1)
    for algo_name in algorithms.keys():
        correct = np.zeros((len(data),n_tgt))
        acc_pred = predictions[algo_name].squeeze(-1)
        for i in range(len(data)):
            for j in range(n_tgt):
                e = min(np.absolute(acc_gt[i,j]-acc_pred[i,:]))
                if e < acc_deg_thres:
                    correct[i,j] = 1
        ACC[algo_name] = 100*np.mean(np.prod(correct,1)).item()

    # compute the MAE and MEDAE
    error = {}
    MAE, MEDAE, = {}, {}

    chamferDist = ChamferDistance()

    for algo_name in algorithms.keys():
        tmp = 0.5*chamferDist(predictions[algo_name],gt,bidirectional=True,reduction="none")
        tmp = tmp.detach().cpu().numpy()
        tmp /= n_tgt
        error[algo_name] = tmp
        MEDAE[algo_name] = np.median(tmp).item()
        MAE[algo_name] = np.mean(tmp).item()

    # save the results
    result = {"error":error,"MAE":MAE,"MEDAE":MEDAE,'predictions':predictions_np,'ACC':ACC,'deg_step':deg_step,'gt':gt.cpu().numpy(),'spatial_spectra':spatial_spectra,'interference_doa':interference_doa, 'freq_range':freq_range}
    scipy.io.savemat(result_path,result)

    # print the results
    print(f"[ Azimuthal DoA estimation setting ] Number of trials: %d, RT60: %.2f seconds, SNR: %.2f dB, SIR: %.2f dB, n_tgt: %d, n_src: %d, n_snapshots: %d, irm_thres: %.2f, acc_deg_thres: %.2f"%(len(data),rt60,snr,sir,n_tgt,n_src,n_snapshots,irm_thres,acc_deg_thres),flush=True)
    for algo_name in algorithms.keys():
        print("{:>22}: {:5.2f}\xb0 (MAE) {:5.2f}\xb0 (MEDAE) {:5.2f}% (ACC)".format(algo_name,MAE[algo_name],MEDAE[algo_name],ACC[algo_name]),flush=True)
    print("[{}] -------------------- Results saved at {}. Elapsed time: {:.2f} seconds".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),result_path,elapsed),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DoA estimation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nsrc', default=2, type=int, help='number of sources including all the targets and interferences')
    parser.add_argument('--ntgt', default=1, type=int, help='number of targets')
    parser.add_argument('--nsnap', default=[i for i in range(20,61,1)], nargs='+', type=int, help='list of number of snapshots')
    parser.add_argument('--rt60s', default=[0.3,0.9], nargs='+', type=float, help='list of RT60')
    parser.add_argument('--snrs', default=[20,10,0], nargs='+', type=int, help='list of SNR')
    parser.add_argument('--sirs', default=[i for i in range(-10,21,2)], nargs='+', type=int, help='list of SIR')
    parser.add_argument('--degstep', default=0.5, type=float, help='1-D grid search resolution (degrees)')
    parser.add_argument('--prefix',default='./results/', type=dir_path, help='path to the result folder')
    parser.add_argument('--algo', default=['DNNthres_MUSIC','DNNthres_Principal','DNNprod_Weighted','DNNprod_EngWeighted'], nargs='+', type=str, help='algorithms to be enabled and evaluated, use All if all algorithms are used')
    parser.add_argument('--model', default='COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny', type=str, help='neural network model to generate the speech ratio mask')
    parser.add_argument('--thres', default=0.9, type=float, help='beta parameter for the binary thresholding (BT)')
    parser.add_argument('--dataset',default='./dataset/', type=dir_path, help='path to the dataset folder')
    parser.add_argument('--device', default='cuda:0', type=check_device, help='specify a CUDA or CPU device, e.g., cuda:0, cuda:1, and cpu')
    parser.add_argument('--model_path',default='./SpeechEnhancement/checkpoint/COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny_loss=L1_mu=0.1_bs=16_nt=3_ni=3_seed=0/last_model.pt', type=str, help='path to the model')
    parser.add_argument('--num_files', default='*', type=str, help='number of files to be evaluated in the dataset, e.g., 50 and 200, or use * if all of the files in the dataset are needed to be evaluated')
    parser.add_argument('--freq_range', default=[50.0,7000.0], nargs='+', type=float, help='frequency band (Hz), this will determine the frequency bins considered in the optimization objective')
    parser.add_argument('--acc_deg_thres', default=3.0, type=float, help='threshold in degrees for computing the accuracy')

    parser_save_spectra_group = parser.add_mutually_exclusive_group()
    parser_save_spectra_group.add_argument('--spectra', dest='spectra', action='store_true', help='save all spatial spectra',default=False)
    parser_save_spectra_group.add_argument('--no-spectra', dest='nospectra', action='store_true', help='do not save all spatial spectra',default=False)

    args = parser.parse_args()

    # check the arguments
    if args.ntgt > args.nsrc:
        raise ValueError("ntgt cannot be larger than nsrc")
    if len(args.freq_range) != 2:
        raise ValueError("freq_range must have two elements")

    save_spectra = args.spectra and not args.nospectra

    num_runs = len(args.nsnap)*len(args.rt60s)*len(args.snrs)*len(args.sirs)
    k = 0
    
    # evaluate the enabled algorithms using the given conditions and dataset
    print("[{}] start getting the DoA results... Number of runs: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),num_runs),flush=True)
    for rt60 in args.rt60s:
        for nsnap in args.nsnap:
            for snr in args.snrs:
                for sir in args.sirs:
                    gt0 = time.time()
                    # run an experiment with C=num_files trials and save the results for the setting
                    getDoAResults(
                            rt60=rt60,
                            snr=snr,
                            sir=sir,
                            n_src=args.nsrc,
                            n_tgt=args.ntgt,
                            model_name=args.model,
                            model_path=args.model_path,
                            device=args.device,
                            result_prefix=args.prefix,
                            n_snapshots=nsnap,
                            deg_step=args.degstep,
                            enabled_algo=args.algo,
                            irm_thres=args.thres,
                            dataset_prefix=args.dataset,
                            num_files=args.num_files,
                            save_spectra=save_spectra,
                            freq_range=args.freq_range,
                            acc_deg_thres=args.acc_deg_thres
                            )
                    gt1 = time.time()
                    k += 1
                    gt_elapsed = gt1 - gt0
                    print("{} to finish...".format(str(datetime.timedelta(seconds=(num_runs-k)*gt_elapsed))))
    print("[{}] completed".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),flush=True)
