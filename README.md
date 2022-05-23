# Towards Alternative Techniques for Improving Adversarial Robustness: Analysis of Adversarial Training at a Spectrum of Perturbations

## Environment (named DL_env) Setup
```
conda create -y -n DL_env python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate DL_env
pip install torchattacks pytorch-msssim scikit-image wand torchmetrics seaborn gdown scipy==1.7.3
```
* Note 1: Each line of each bash file below executes backround process to run code in parallel. Carefull comment out code in bash file based on the GPUs avaialble in your machine
* Note 2: If you'd like to skip training models, please download the 'saved_state_dicts' folder from [this drive link](https://drive.google.com/drive/folders/13D-3gLl8_SAvnoWUf8LNTdc54XI1lx5y?usp=sharing) and proceed to the Perturbation Spectrum Analysis section

## Training Models
### Standard and Adversarial Training
```
bash train.sh 
```

### Incremental Adversarial Training (See Appendix A.2)
```
bash wait_and_train_incrementally.sh
```

## Perturbation Spectrum Analysis and New Techniques
### Section 3 (Overdesigning for Robust Generalization)
```
bash Section3.sh
python which_of_them_generalizes_best_plotter.py
```

### Section 4 (Intermediate Feature Quantization)
```
bash Section4_Transfer_PGD.sh
python quantize_all_channels_tabler.py

bash Section4_BPDA.sh
python quantize_all_channels_tabler_BPDA.py
```

### Section 5 (AT and Norm of CNN Kernels)
```
bash Section5.sh 
bash Section5_plot.sh 
```

### Section 6 (Training with Larger Perturbations and Common-Corruptions)
```
bash Section6_part1.sh 
python test_on_corruptions_tabler.py

bash Section6_part2.sh 
python test_firing_on_corruptions_plotter.py --model-name resnet18
python test_firing_on_corruptions_plotter.py --model-name wrn
```

