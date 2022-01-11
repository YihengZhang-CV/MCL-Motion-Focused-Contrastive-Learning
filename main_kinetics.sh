#!/bin/bash
# data
root_path='/path/to/data'
root_path_flow='/path/to/data'
root_path_mag='/path/to/data'
list_path='data/kinetics/train_list.csv'
list_path_test='data/kinetics/val_list.csv'
dataset='kinetics'

# model
pretrained_model='/path/to/model'

# gpu
gpus=4

# optimizer
lr=0.02
epochs=400

# train
python3 -m torch.distributed.launch --nproc_per_node $gpus train.py  \
--root-path $root_path --list-path $list_path \
--root-path-flow $root_path_flow --root-path-mag $root_path_mag --dataset $dataset \
--epochs $epochs --base-learning-rate $lr \
--pretrained-model $pretrained_model 

# linear-probe evaluation (train a fc)
python3 -m torch.distributed.launch --nproc_per_node $gpus evaluate/linear_probe/eval_fc.py  \
--root-path $root_path --list-path $list_path --list-path-test $list_path_test --dataset $dataset  \
--trainval 'train'

python3 -m torch.distributed.launch --nproc_per_node $gpus evaluate/linear_probe/eval_fc.py  \
--root-path $root_path --list-path $list_path --list-path-test $list_path_test --dataset $dataset  --pretrained-model 'output/eval_output_linear/checkpoints/current.pth' \
--trainval 'test'
