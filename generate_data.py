"""
Created on Sat Mar 5 2022
@author: Kuan-Lin Chen
"""
import os
import glob
import random
import time
import numpy as np
import pyroomacoustics as pra
import librosa
from datetime import datetime

def cart2sph(cart):
    x = cart[:,0]
    y = cart[:,1]
    z = cart[:,2]
    rxy = np.hypot(x, y)
    r = np.hypot(rxy, z)
    theta = np.arctan2(rxy, z)
    phi = np.arctan2(y, x)
    theta = np.where(theta<0,theta+2*np.pi,theta)
    phi = np.where(phi<0,phi+2*np.pi,phi)
    sph = np.stack((r,theta,phi),axis=1)
    return sph

def overlap(phi,phi_list,min_sep_rad):
    for p in phi_list:
        ae = abs(phi-p)
        d = min(ae,2*np.pi-ae)
        if d < min_sep_rad:
            return True
    return False

def getArrayCenter(mic_locs):
    center = [j/len(mic_locs) for j in [sum(i) for i in zip(*mic_locs)]]
    return center

def getAudioList(pattern):
    paths = glob.glob(pattern)
    return paths

def generateRoomData(
                    src_wav_paths,
                    seed,
                    path = 'room_data',
                    fs = 16000,
                    rt60 = 0.2,
                    room_dim = [10,7.5,3.5],
                    ray_tracing = False,
                    air_absorption = False,
                    src_locs = [[1,1,1],[2,5,2.5],[9,5,2]],
                    mic_locs = [[5,5,2],[5,6,2],[5,7,2]],
                    snr = 30,
                    sir = 10,
                    n_tgt = 2
                    ):
    # set the random seed
    np.random.seed(seed)
    n_src = len(src_locs)
    if n_src != len(src_wav_paths):
        raise AssertionError("the number of elements of src_wav_paths and src_locs should be equivalent")
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    # Create the room
    room = pra.ShoeBox(room_dim,fs=fs,materials=pra.Material(e_absorption),max_order=max_order,ray_tracing=ray_tracing,air_absorption=air_absorption)

    # Activate the ray tracing
    if ray_tracing is True:
        room.set_ray_tracing()

    # place the source in the room
    for i in range(n_src):
        audio = librosa.load(src_wav_paths[i],sr=fs,mono=True)[0]
        room.add_source(src_locs[i], signal=audio, delay=0)

    # define the locations of the microphones and place the array in the room
    room.add_microphone_array(np.c_[tuple(mic_locs)])

    # compute the center of the array
    center = getArrayCenter(mic_locs)
    # compute the spherical coordinate, i.e., the length r, the polar angle theta, the azimuthal angle phi
    cart = np.asarray(src_locs) - np.asarray(center)
    sph = cart2sph(cart).tolist() 

    # the extra arguments are given in a dictionary
    callback_mix_kwargs = {'snr':snr,'sir':sir,'n_src':n_src,'n_tgt':n_tgt,'ref_mic':0}

    def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):
        # first normalize all separate recording to have unit power at microphone one
        p_mic_ref = np.std(premix[:,ref_mic,:], axis=1)
        premix /= p_mic_ref[:,None,None]

        # now compute the power of interference signal needed to achieve desired SIR
        if n_tgt < n_src:
            sigma_i = np.sqrt(10 ** (- sir / 10))
            premix[n_tgt:n_src,:,:] *= sigma_i

        # compute noise variance
        sigma_n = np.sqrt(10 ** (- snr / 10))

        # Mix down the recorded signals
        mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])
        return mix

    # Run the simulation (this will also build the RIR automatically)
    room.simulate(callback_mix=callback_mix,callback_mix_kwargs=callback_mix_kwargs)
    mix = room.mic_array.signals
    c = np.abs(mix).max()
    mix = mix / c
    # save the setting
    state = {'fs':fs,'rt60':rt60,'room_dim':room_dim,'ray_tracing':ray_tracing,'air_absorption':air_absorption,'src_locs':src_locs,'src_wav_paths':src_wav_paths,'snr':snr,'sir':sir,'n_tgt':n_tgt,'n_src':n_src,'center':center,'sph':sph,'mix':mix}
    np.save(path+".npy",state)

def generateRoomDataset(tgt_wav_paths,
                        int_wav_paths,
                        base_path = 'dataset',
                        N = 200,
                        fs = 16000,
                        rt60_list = [0.3,0.9],
                        rd_u = [9,7,3.5],
                        rd_l = [9,7,3.5],
                        sr_lu = [1.0,3.0],
                        sh_lu = [1.0,1.8],
                        min_sep_deg = 10,
                        snr_list = [20,10,0],
                        sir_list = [i for i in range(-10,21,2)],
                        n_srcs = [2,3],
                        n_tgts = [1,1],
                        mic_center = [4.5,3.5,1.75],
                        mic_arr = [[ 0.02,  0.0, 0.0],
                                    [ 0.0,  0.02, 0.0],
                                    [-0.02,  0.0, 0.0],
                                    [ 0.0, -0.02, 0.0],
                                    [ 0.02, 0.02, 0.0],
                                    [-0.02,-0.02, 0.0],
                                    [ 0.02,-0.02, 0.0],
                                    [-0.02, 0.02, 0.0],
                                    [ 0.00, 0.00, 0.0]]
                        ):
    # check the number of srouce wave files provided
    if len(tgt_wav_paths) < max(n_tgts):
        raise AssertionError("the number of target wave file paths should be larger than or equal to each elements in n_tgts")
    if len(int_wav_paths) < max([n_srcs[i]-n_tgts[i] for i in range(len(n_tgts))]):
        raise AssertionError("the number of interference wave file paths should be larger than or equal to the number of interferences")
    if sr_lu[0]>sr_lu[1]:
        raise AssertionError("the lower and upper bounds of the source radius is incorrect, the sr_lu[0] should be smaller than or equal to sr_lu[1]")
    if min_sep_deg <=0 or min_sep_deg >= 180:
        raise AssertionError("0 <= min_sep_deg <= 180")
    if len(n_srcs) > 360/min_sep_deg:
        raise AssertionError("the number of sources is too large, it should be smaller than or equal to 360/min_sep_deg") 
    # create the dataset folder
    if not os.path.exists(base_path):
            os.mkdir(base_path)
    else:
        raise FileExistsError("the folder already exists, try to use another base_path")
    # convert to radian
    min_sep_rad = min_sep_deg*np.pi/180
    # margins
    mic_margin = 0.1
    wm = 0.3 # wall margin
    # compute the center and radius of the array
    center = np.asarray(mic_center)
    arr = np.asarray(mic_arr)
    mic_locs = (arr+center).tolist()
    r = max(np.linalg.norm(arr,axis=1))
    # check the position and the size of the array
    if not (np.all(center + r + mic_margin + wm < rd_l) and np.all(r + mic_margin + wm < center)):
        raise AssertionError("the entire array should be sufficiently covered in the room")
    # check the position of the sources
    if r + mic_margin >= sr_lu[0]:
        raise AssertionError("the distance between the source and the array should be sufficiently large")
    if not (center[0] + sr_lu[1] + wm < rd_l[0] and center[0] - sr_lu[1] > wm and center[1] + sr_lu[1] + wm < rd_l[1] and center[1] - sr_lu[1] > wm):
        raise AssertionError("the distance between the source and the wall should be sufficiently large")
    # create a dataset with potentially different numbers of settings, i.e., (# of sources and # of targets)
    setting = {'fs':fs,'mic_locs':mic_locs,'snr_list':snr_list,'sir_list':sir_list,'rt60_list':rt60_list,'N':N,'min_sep_deg':min_sep_deg}
    np.save(base_path+"/setting.npy",setting)
    for k in range(len(n_srcs)):
        for i in range(N):
            t0 = time.time()
            random.seed(i)
            room_dim = [random.uniform(rd_l[0],rd_u[0]),random.uniform(rd_l[1],rd_u[1]),random.uniform(rd_l[2],rd_u[2])]
            src_locs = []
            phi_list = []
            for j in range(n_srcs[k]):
                phi = random.uniform(0,2*np.pi)
                while overlap(phi,phi_list,min_sep_rad):
                    phi = random.uniform(0,2*np.pi)
                phi_list.append(phi)
            for j in range(n_srcs[k]):
                s_r = random.uniform(sr_lu[0],sr_lu[1])
                phi = phi_list[j]
                s_x = s_r*np.cos(phi) + center[0]
                s_y = s_r*np.sin(phi) + center[1]
                s_z = random.uniform(sh_lu[0],sh_lu[1])
                src_locs.append([s_x,s_y,s_z])
            src_wav_paths = random.sample(tgt_wav_paths,n_tgts[k])+random.sample(int_wav_paths,n_srcs[k]-n_tgts[k])
            for rt60 in rt60_list:
                for snr in snr_list:
                    for sir in sir_list:
                        path = '{}/nsrc={}_ntgt={}_rt60={}_snr={}_sir={}'.format(base_path,n_srcs[k],n_tgts[k],rt60,snr,sir)
                        path_i = path+'/'+str(i)
                        if not os.path.exists(path):
                            os.mkdir(path)
                        generateRoomData(path = path_i,
                                         seed = i,
                                         fs = fs,
                                         rt60 = rt60,
                                         room_dim = room_dim,
                                         src_locs = src_locs,
                                         mic_locs = mic_locs,
                                         src_wav_paths = src_wav_paths,
                                         snr = snr,
                                         sir = sir,
                                         n_tgt = n_tgts[k])
                        print("{} A recording has been created at {} with SNR={:.2f}, SIR={:.2f}, rt60={:.2f}, and room_dim=({:.2f},{:.2f},{:.2f})".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),path_i,snr,sir,rt60,room_dim[0],room_dim[1],room_dim[2]),flush=True)
            t1 = time.time()
            print("{} Recordings for all rt60, snr, and sir have been completed for i/N: {}/{} under n_src={} and n_tgt={} | elapsed: {:5.1f} s".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i,N,n_srcs[k],n_tgts[k],t1-t0),flush=True)
        print("{} {} datapoints have been generated for n_src={} and n_tgt={}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),N,n_srcs[k],n_tgts[k]),flush=True)

if __name__ == '__main__':
    generateRoomDataset(tgt_wav_paths = getAudioList("./../Datasets/timit_lowercase/test/*/*/*.wav"),
                        int_wav_paths = getAudioList("./../Datasets/Nonspeech/test/*.wav"))
