mkdir logs/

GPUs=0

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u ST.py --tqdm-off > logs/ST_resnet18.log &

for EPS in 1 2 3 4 5 6 7 8 9 10 11 12
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u PGDAT.py --epsilon ${EPS} --epsilon-test 2 --tqdm-off > logs/PGDAT_${EPS}.log &
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u ST.py --tqdm-off --model-name wrn --batch-size 128 > logs/ST_wrn.log &

for EPS in 2 4 6 8 10 12
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u PGDAT.py --batch-size 128 --model wrn --epsilon ${EPS} --epsilon-test 2 --tqdm-off > logs/PGDAT_wide_${EPS}.log &
done