#!/bin/bash
# data
root_path='/path/to/data'
root_path_flow='/path/to/data'
root_path_mag='/path/to/data'
list_path='data/ucf101/train_list_01.csv'
list_path_test='data/ucf101/test_list_01.csv'
dataset='ucf101'

# model
pretrained_model='/path/to/model'

# gpu
gpus=4

# optimizer
lr=0.01
epochs=200

# train
python3 -m torch.distributed.launch --nproc_per_node $gpus train.py  \
--root-path $root_path --list-path $list_path  \
--root-path-flow $root_path_flow --root-path-mag $root_path_mag --dataset $dataset \
--epochs $epochs --base-learning-rate $lr \
--pretrained-model $pretrained_model

# linear-probe evaluation (linear SVM)
python3 -m torch.distributed.launch --nproc_per_node $gpus evaluate/linear_probe/eval_svm_feature_extract.py  \
--root-path $root_path --list-path $list_path --dataset $dataset --mode 'trainsvm'
python3 -m torch.distributed.launch --nproc_per_node $gpus evaluate/linear_probe/eval_svm_feature_extract.py  \
--root-path $root_path --list-path $list_path_test --dataset $dataset --mode 'test'

python3 evaluate/linear_probe/eval_svm_feature_perf.py