mkdir logs
GPUs=0

until [ -f ./saved_state_dicts/ST_resnet18/ep_19.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 2 > logs/StackedPGDAT_resnet18_2.log &


until [ -f ./saved_state_dicts/StackedPGDAT_resnet18_2.0/ep_76.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 4 > logs/StackedPGDAT_resnet18_4.log &


until [ -f ./saved_state_dicts/StackedPGDAT_resnet18_4.0/ep_76.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 6 > logs/StackedPGDAT_resnet18_6.log &

until [ -f ./saved_state_dicts/StackedPGDAT_resnet18_6.0/ep_76.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 8 > logs/StackedPGDAT_resnet18_8.log &

until [ -f ./saved_state_dicts/StackedPGDAT_resnet18_8.0/ep_76.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 10 > logs/StackedPGDAT_resnet18_10.log &

until [ -f ./saved_state_dicts/StackedPGDAT_resnet18_10.0/ep_76.pth ]
do
    sleep 10
done

CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u StackedPGDAT.py --batch-size 512 --num-epochs 100 --model-name resnet18 --epsilon 12 > logs/StackedPGDAT_resnet18_12.log &
