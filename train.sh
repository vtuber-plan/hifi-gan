MASTER_ADDR=localhost\
MASTER_PORT=8018\
WORLD_SIZE=4\
NODE_RANK=0\
LOCAL_RANK=0\
python train.py --accelerator 'gpu' --config ./configs/24k.json --device 0 --num-nodes 4