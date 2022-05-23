mkdir logs

GPUs=0

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --model-type Stacked > logs/eigenvalue_diffs_hist_Stacked.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off > logs/eigenvalue_diffs_hist_normal.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --model wrn --tqdm-off > logs/eigenvalue_diffs_hist_wrn.log &

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --model-type Stacked --metric mean > logs/eigenvalue_diffs_hist_Stacked2.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --tqdm-off --metric mean > logs/eigenvalue_diffs_hist_normal2.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u eigenvalue_difference_histograms.py --model wrn --tqdm-off --metric mean > logs/eigenvalue_diffs_hist_wrn2.log &


