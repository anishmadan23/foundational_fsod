#!/bin/sh

# CUDA_VISIBLE_DEVICES=0 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_0/ --add_name tiny &
CUDA_VISIBLE_DEVICES=1 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_0/ --add_name large &

# CUDA_VISIBLE_DEVICES=2 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_1/ --add_name tiny &
CUDA_VISIBLE_DEVICES=3 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_1/ --add_name large &

# CUDA_VISIBLE_DEVICES=4 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_2/ --add_name tiny &
CUDA_VISIBLE_DEVICES=5 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_seed_2/ --add_name large &

# CUDA_VISIBLE_DEVICES=6 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_0/ --add_name tiny &
CUDA_VISIBLE_DEVICES=7 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_0/ --add_name large 


# CUDA_VISIBLE_DEVICES=0 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_1/ --add_name tiny &
CUDA_VISIBLE_DEVICES=0 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_1/ --add_name large &

# CUDA_VISIBLE_DEVICES=2 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_2/ --add_name tiny &
CUDA_VISIBLE_DEVICES=2 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_2/ --add_name large &

# CUDA_VISIBLE_DEVICES=4 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_0/ --add_name tiny &
CUDA_VISIBLE_DEVICES=4 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_0/ --add_name large &

# CUDA_VISIBLE_DEVICES=6 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_1/ --add_name tiny &
CUDA_VISIBLE_DEVICES=6 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed1_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_1/ --add_name large 

# CUDA_VISIBLE_DEVICES=0 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_2/ --add_name tiny &
CUDA_VISIBLE_DEVICES=1 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_seed2_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_seed_2/ --add_name large 






### best split
# CUDA_VISIBLE_DEVICES=0 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_best_split/ --add_name tiny &
# CUDA_VISIBLE_DEVICES=1 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_5shot/nuimages/nuimages_5_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 5 --save_path MODEL/nuimages_fsod_5_shots_best_split/ --add_name large &

# CUDA_VISIBLE_DEVICES=2 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_best_split/ --add_name tiny &
# CUDA_VISIBLE_DEVICES=3 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_best_split/ --add_name large &

# CUDA_VISIBLE_DEVICES=4 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-t-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_best_split/ --add_name tiny &
# CUDA_VISIBLE_DEVICES=5 python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_30shot/nuimages/nuimages_30_shots_best_split_fsod.yaml --dataset nuim --num_vision_queries 30 --save_path MODEL/nuimages_fsod_30_shots_best_split/ --add_name large 
