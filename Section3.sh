mkdir logs/

GPUs=0

#### ResNet18 ######

for EPS in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u which_of_them_generalizes_best.py --epsilon-test ${EPS} --tqdm-off > logs/which_of_them_generalizes_best_${EPS}by255.log &
done


#### WRN ######

for EPS in 0 2 4 6 8 10 12
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u which_of_them_generalizes_best.py --model-name wrn --batch-size 128 --epsilon-test ${EPS} --tqdm-off > logs/wrn_which_of_them_generalizes_best_${EPS}by255.log &
done
