# train config
list_file=dataset/hmdb51_frame128_train_list.txt
root_path=/path/to/data
num_classes=51
ra_n=3
ra_m=17
gpu_num=4

# eval config
eva_list_file=dataset/hmdb51_frame128_test_list.txt
eva_root_path=/path/to/data
pretrained_model=../../output/checkpoints/current.pth


# train
python3 -m torch.distributed.launch --nproc_per_node=$gpu_num train_3d.py \
--list-file=$list_file \
--root-path=$root_path \
--num-classes=$num_classes \
--pretrained-model=$pretrained_model \
--ra-n $ra_n \
--ra-m $ra_m

# extract feature first
python3 -m torch.distributed.launch --nproc_per_node=$gpu_num extract_score_3d.py \
--list-file=$eva_list_file \
--root-path=$eva_root_path \
--pretrained-model=../../output/eval_output_finetune/current.pth \
--num-classes=$num_classes \
--crop-idx=0

python3 -m torch.distributed.launch --nproc_per_node=$gpu_num extract_score_3d.py \
--list-file=$eva_list_file \
--root-path=$eva_root_path \
--pretrained-model=../../output/eval_output_finetune/current.pth \
--num-classes=$num_classes \
--crop-idx=1

python3 -m torch.distributed.launch --nproc_per_node=$gpu_num extract_score_3d.py \
--list-file=$eva_list_file \
--root-path=$eva_root_path \
--pretrained-model=../../output/eval_output_finetune/current.pth \
--num-classes=$num_classes \
--crop-idx=2

# eval
python3 merge_score.py \
--num-classes=$num_classes \
--num-gpu=$gpu_num \
--list-file=$eva_list_file