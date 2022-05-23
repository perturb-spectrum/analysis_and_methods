mkdir logs

GPUs=0

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u test_on_corruptions.py --tqdm-off > logs/normal_corruptions.log &
# CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u test_on_corruptions.py --tqdm-off --model-type Stacked > logs/Stacked_corruptions.log &

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u test_on_corruptions.py --model-name wrn --tqdm-off > logs/wrn_normal_corruptions.log &
