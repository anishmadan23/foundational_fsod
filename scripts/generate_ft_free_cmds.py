import glob
import os

# change these

def get_addn_cfg_file(shots, seed):
    return f'configs/vision_query_{shots}shot/nuimages/nuimages_{shots}_shots_seed{seed}_fsod.yaml'

def get_vq_bank(shots, seed, model_suffix):
    return f'MODEL/nuimages_fsod_{shots}_shots_seed_{seed}/nuim_fsod_query_{shots}_pool7_sel_{model_suffix}.pth'

def get_model_weight(model_suffix):
    if model_suffix=='tiny':
        return f'MODEL/mq-glip-t'
    elif model_suffix=='large':
        return f'MODEL/mq-glip-l'
    else:
        raise ValueError(f'Unknown model suffix {model_suffix}')

def get_base_cfg_file(model_suffix):
    if model_suffix=='tiny':
        return 'configs/pretrain/my_configs/mq-glip-t-nuim.yaml'
    elif model_suffix=='large':
        return 'configs/pretrain/my_configs/mq-glip-l-nuim.yaml'
    else:
        raise ValueError(f'Unknown model suffix {model_suffix}')

cur_gpu=0
all_cmds = []

num_seeds = 3
# num_shots = [5,10,30]
num_shots = [30]
model_suffixes = ['tiny', 'large']
num_gpus=8
visible_devices = '0,1,2,3,4,5,6,7'
vnice=False
modes = ['t','v','tv']   # text only, vision only, text+vision

num_total_expts = len(num_shots)*num_seeds*len(model_suffixes)*len(modes)
idx = 0
# base_config_file_path = 'configs/pretrain/my_configs/mq-glip-t-nuim.yaml'
for shots in num_shots:
    for seed in range(num_seeds):
        for model_type in model_suffixes:
            training_cmd = f'CUDA_VISIBLE_DEVICES={visible_devices} python -m torch.distributed.launch --nproc_per_node {num_gpus} tools/test_grounding_net.py --config-file {get_base_cfg_file(model_type)} --additional_model_config {get_addn_cfg_file(shots, seed)} VISION_QUERY.QUERY_BANK_PATH {get_vq_bank(shots, seed, model_type)} MODEL.WEIGHT {get_model_weight(model_type)} TEST.IMS_PER_BATCH {num_gpus} VISION_QUERY.NUM_QUERY_PER_CLASS {shots} VISION_QUERY.MAX_QUERY_NUMBER {shots} DATASETS.FEW_SHOT {shots}'
            for mode in modes:
                if mode=='t': # text only
                    cur_output_dir = f'results/nuimages_fsod/{shots}_shots_seed_{seed}/model_{model_type}_text_only/'
                    os.makedirs(cur_output_dir, exist_ok=True)
                    new_training_cmd = training_cmd + f' OUTPUT_DIR {cur_output_dir} VISION_QUERY.ENABLED False'
                elif mode=='v': # vision only
                    cur_output_dir = f'results/nuimages_fsod/{shots}_shots_seed_{seed}/model_{model_type}_vision_only/'
                    os.makedirs(cur_output_dir, exist_ok=True)
                    new_training_cmd = training_cmd + f' OUTPUT_DIR {cur_output_dir} VISION_QUERY.MASK_DURING_INFERENCE True VISION_QUERY.TEXT_DROPOUT 1.0'
                elif mode=='tv':
                    cur_output_dir = f'results/nuimages_fsod/{shots}_shots_seed_{seed}/model_{model_type}_text_and_vision/'
                    os.makedirs(cur_output_dir, exist_ok=True)
                    new_training_cmd = training_cmd + f' OUTPUT_DIR {cur_output_dir}'
                else:
                    raise ValueError(f'Unknown mode {mode}')
                idx+=1
                # cur_gpu+=1
                all_cmds.append(new_training_cmd)

with open(f'./scripts/nuim_test_cmds_{num_shots[0]}_shots.sh', 'w') as f:
    f.write("#!/bin/bash \n")
    for cmd in all_cmds:
        f.write(cmd+'\n')



# Sample command
# python -m torch.distributed.launch --nproc_per_node 8   tools/test_grounding_net.py \                                                                            SIGINT(2) ↵  10033  15:12:41
# --config-file configs/pretrain/my_configs/mq-glip-t-nuim.yaml \
# --additional_model_config configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml \
# VISION_QUERY.QUERY_BANK_PATH MODEL/nuimages_fsod_10_shots_seed_0/nuim_fsod_query_10_pool7_sel_tiny.pth \
# MODEL.WEIGHT MODEL/mq-glip-t \
# TEST.IMS_PER_BATCH 8 OUTPUT_DIR results/nuimages/10_shots_seed_0/ VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10