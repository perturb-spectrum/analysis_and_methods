mkdir logs

GPUs=0

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --plot --model-type Stacked > logs/eigenvalue_diffs_hist_Stacked.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --plot > logs/eigenvalue_diffs_hist_normal.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --model wrn --tqdm-off --plot > logs/eigenvalue_diffs_hist_wrn.log &

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --plot --model-type Stacked --metric mean > logs/eigenvalue_diffs_hist_Stacked.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --plot --metric mean > logs/eigenvalue_diffs_hist_normal.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --model wrn --tqdm-off --plot --metric mean > logs/eigenvalue_diffs_hist_wrn.log &


