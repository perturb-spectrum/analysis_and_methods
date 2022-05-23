mkdir logs

until [ -f ./saved_state_dicts/ST_wrn/ep_18.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 2 > logs/StackedPGDAT_wrn_2.log &


until [ -f ./saved_state_dicts/StackedPGDAT_wrn_2.0/ep_80.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 4 > logs/StackedPGDAT_wrn_4.log &


until [ -f ./saved_state_dicts/StackedPGDAT_wrn_4.0/ep_80.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 6 > logs/StackedPGDAT_wrn_6.log &

until [ -f ./saved_state_dicts/StackedPGDAT_wrn_6.0/ep_80.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 8 > logs/StackedPGDAT_wrn_8.log &

until [ -f ./saved_state_dicts/StackedPGDAT_wrn_8.0/ep_80.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 10 > logs/StackedPGDAT_wrn_10.log &

until [ -f ./saved_state_dicts/StackedPGDAT_wrn_10.0/ep_80.pth ]
do
    sleep 10
done

nohup python -u StackedPGDAT.py --model wrn --batch-size 256 --num-epochs 80 --model-name wrn --epsilon 12 > logs/StackedPGDAT_wrn_12.log &
