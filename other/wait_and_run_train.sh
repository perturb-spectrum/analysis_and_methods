until [ -f ./saved_state_dicts/PGDAT_resnet18_5.0/ep_100.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=0 nohup python -u PGDAT.py --epsilon 12 --epsilon-test 2 --tqdm-off > logs/PGDAT_12.log &

until [ -f ./saved_state_dicts/PGDAT_resnet18_6.0/ep_100.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=0 nohup python -u PGDAT.py --epsilon 13 --epsilon-test 2 --tqdm-off > logs/PGDAT_13.log &

until [ -f ./saved_state_dicts/PGDAT_resnet18_9.0/ep_100.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=0 nohup python -u PGDAT.py --epsilon 14 --epsilon-test 2 --tqdm-off > logs/PGDAT_14.log &

