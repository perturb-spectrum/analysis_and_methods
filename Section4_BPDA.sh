
mkdir logs

GPUs=0

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 2 --tqdm-off > logs/BPDA_full_scale_8_eps_test_2.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 4 --tqdm-off > logs/BPDA_full_scale_8_eps_test_4.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 6 --tqdm-off > logs/BPDA_full_scale_8_eps_test_6.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 8 --tqdm-off > logs/BPDA_full_scale_8_eps_test_8.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 10 --tqdm-off > logs/BPDA_full_scale_8_eps_test_10.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale 8 --epsilon-test 12 --tqdm-off > logs/BPDA_full_scale_8_eps_test_12.log &


CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 2 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_2.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 4 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_4.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 6 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_6.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 8 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_8.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 10 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_10.log &
CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale 8 --epsilon-test 12 --tqdm-off > logs/wrn_BPDA_full_scale_8_eps_test_12.log &


# ############# ResNet18 Ablations ############
# for SCALE in 4 6 10 12
# do
#     for DELTA in 2 4 6 8 10 12
#     do
#         CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --scale ${SCALE} --epsilon-test ${DELTA} --tqdm-off > logs/BPDA_full_scale_${SCALE}_eps_test_${DELTA}.log &
#     done
# done

# ############## WRN Ablations ############
# for SCALE in 4 6 10 12
# do
#     for DELTA in 2 4 6 8 10 12
#     do
#         CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u quantize_all_channels_BPDA.py --model-name wrn --batch-size 128 --scale ${SCALE} --epsilon-test ${DELTA} --tqdm-off > logs/wrn_BPDA_full_scale_${SCALE}_eps_test_${DELTA}.log &
#     done
# done