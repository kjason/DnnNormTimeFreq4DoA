# A DNN based Normalized Time-Frequency Weighted Criterion for Robust Wideband DoA Estimation

This repository is the official implementation of the paper accepted at ICASSP 2023, [A DNN BASED NORMALIZED TIME-FREQUENCY WEIGHTED CRITERION FOR ROBUST WIDEBAND DOA ESTIMATION](https://arxiv.org/abs/2302.10147).

- Download the paper from IEEE Xplore or [arXiv](https://arxiv.org/abs/2302.10147).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Please install Python before running the above setup command. The code was tested on Python 3.8.10.

## Training a DNN model to estimate time-frequency weights
Before training, one may need to download the speech and nonspeech datasets and specify the "target_pattern" and "interference_pattern" in "data.py" so as to load the target data and interference data properly. Please see Section 5 of the paper for more details.

To learn the time-frequency weights, we train a DNN model to estimate the ideal ratio mask from noisy speech. Only single-channel speech and nonspeech corpora are required. Note that this part is independent of the DoA estimation methods. First, we go to the directory "SpeechEnhancement" as follows:
```
cd SpeechEnhancement
```
Then, we train a DNN model by the following command:
```train
python main.py
```
All the training results including all the hyperparameters, training curves, parameters will be saved to the checkpoint directory. One can also stop training at any time and continue the training process from the last saved checkpoint by using the resume option "--resume" (default is true).

One can customize the hyperparameters by specifying the options. For example, one can run:
```train_options_example
python main.py --batch_size 8 --mu 0.05 --loss MSE --model COMPLEX_IRM_Sigmoid_U_Net_Small --checkpoint_folder my-new-experiment
```

Run this command to see how to customize your training:
```train_options
python main.py --help
```

To evaluate the performance, run
```train_eval
python eval.py
```

## Data for evaluating DoA estimation methods

One can create different simulation scenarios and store them in different folders for research. One can modify the settings in the script "generate_data.py" for a particular simulation scenario. To generate data for the evaluation of different DoA estimation algorithms in the paper, run:
```eval_data
python generate_data.py
```
Regarding the setting of the room model, microphone array, and the sound sources, please see Section 5 of the paper for more details.

## Chamfer distance

To measure the performance of DoA estimation for multiple targets, the [chamferdist repo](https://github.com/krrish94/chamferdist) has been modified to use modular arithmetic and L1 norm in computing the distance of every pair of two points. Such a modification replaces the squared L2 norm. The modular arithmetic is used to compute the distance between two azimuthal angles.

> Note that the above Chamfer distance reduces to the modular L1 distance when there is only a single target. If one is only interested in the single target case, then one can replace the Chamfer distance with a valid implementation of modular L1 norm.

Please download the chamferdist repo in the link above and replace the files knn_cpu.cpp, knn.cu, and chamfer.py with the files in the directory "ChamferDistance" in this repo. Note that these files were copied from the [chamferdist repo](https://github.com/krrish94/chamferdist) and slightly modified to fit in our application. Also note that the implementation in [chamferdist repo](https://github.com/krrish94/chamferdist) was repackaged from [pytorch3d](https://github.com/facebookresearch/pytorch3d).

Then, build the chamferdist from source. Please see the instructions in the [chamferdist repo](https://github.com/krrish94/chamferdist).

The implementation of the Chamfer distance in the original chamferdist repo may have been changed, but it is easy for one to modify the abvoe-mentioned files in a few lines to use modular arithmetic and L1 norm.

## Evaluation

To evaluate DoA estimation algorithms on the conditions RT60 0.3s, SNR 20 dB, SIR -6, 0, 6 dB and number of snapshots 50, run the following command:
```eval_methods
python doa.py --nsrc 2 --ntgt 1 --rt60s 0.3 --snrs 20 --prefix ./results/ --algo DNNthres_MUSIC DNNthres_Principal DNNprod_Weighted DNNprod_EngWeighted --nsnap 50 --sirs -6 0 6 --dataset ./dataset/
```
Note that there are two sources in total and there is only one target source in the above command. The number of sources "nsrc" is the sum of the number of target sources and the number of interference sources. This command will create a directory named "results" and save all the results therein. The dataset used by the DoA estimation algorithms is given by the directory "dataset" in the above command. One can use the option "--algo" to specify different DoA estimation algorithms. The above example only evaluates 4 methods in the same time but one is allowed to specify more. Please see the "doa.py" for more details on available algorithms.

Run this command to see how to customize your evaluation:
```options
python doa.py --help
```
Please see Section 5 of the paper for more details on the evaluation metrics.

## BibTeX
```
@inproceedings{chen2023dnn,
  title={A {DNN} based Normalized Time-Frequency Weighted Criterion for Robust Wideband {DoA} Estimation},
  author={Chen, Kuan-Lin and Lee, Ching-Hua and Rao, Bhaskar D. and Garudadri, Harinath},
  booktitle={International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```
