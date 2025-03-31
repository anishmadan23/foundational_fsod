import os
import numpy as np
import csv

def main(datasets_links_filepath, cmds_script_path, num_gpus=7, text=True, vision=False, dataset_type="rf20vl_fsod"):
    dataset_names = []
    with open(datasets_links_filepath, "r") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx>0: # skip header
                url = row[0]
                dataset_names.append(url.split('/')[-2])
    
    with open(cmds_script_path, 'w') as bash_script:
        
        bash_script.write("#!/bin/bash\n\n")
        
        for dataset in dataset_names:
            # updated_dataset_name = dataset.replace("-", "_")
            if text and vision:
                command = f"""python -m torch.distributed.launch --nproc_per_node {num_gpus} tools/test_grounding_net.py --config-file configs/pretrain/mq-glip-l.yaml --additional_model_config configs/rf_configs/{dataset}/10_shots/multimodal_prompting.yaml VISION_QUERY.QUERY_BANK_PATH MODEL/rf_data/{dataset}_fsod_best_split/{dataset}_query_10_sel_large.pth MODEL.WEIGHT MODEL/glip_large_model.pth TEST.IMS_PER_BATCH {num_gpus} VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR /data3/anishmad/msr_thesis/rf_fsod_baselines/mqdet_results/{dataset_type}/{dataset}/model_large_text_and_vision/  \n"""
            elif text:
                command = f"""python -m torch.distributed.launch --nproc_per_node {num_gpus} tools/test_grounding_net.py --config-file configs/pretrain/mq-glip-l.yaml --additional_model_config configs/rf_configs/{dataset}/10_shots/multimodal_prompting.yaml VISION_QUERY.QUERY_BANK_PATH MODEL/rf_data/{dataset}_fsod_best_split/{dataset}_query_10_sel_large.pth MODEL.WEIGHT MODEL/glip_large_model.pth TEST.IMS_PER_BATCH {num_gpus} VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR /data3/anishmad/msr_thesis/rf_fsod_baselines/mqdet_results/{dataset_type}/{dataset}/model_large_text_only/  VISION_QUERY.ENABLED False \n"""
            elif vision:
                command = f"""python -m torch.distributed.launch --nproc_per_node {num_gpus} tools/test_grounding_net.py --config-file configs/pretrain/mq-glip-l.yaml --additional_model_config configs/rf_configs/{dataset}/10_shots/multimodal_prompting.yaml VISION_QUERY.QUERY_BANK_PATH MODEL/rf_data/{dataset}_fsod_best_split/{dataset}_query_10_sel_large.pth MODEL.WEIGHT MODEL/glip_large_model.pth TEST.IMS_PER_BATCH {num_gpus} VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR /data3/anishmad/msr_thesis/rf_fsod_baselines/mqdet_results/{dataset_type}/{dataset}/model_large_vision_only/ VISION_QUERY.MASK_DURING_INFERENCE True VISION_QUERY.TEXT_DROPOUT 1.0 \n"""
            else:
                raise ValueError("At least one of text or vision must be True.")
            # Write the command to the bash script
            bash_script.write(command)

if __name__=='__main__':
    datasets_links_filepath = 'datasets_links.csv'
    dataset_type="rf20vl_fsod"     # rf100vl_fsod
    cmds_script_path_model_large_text_only = 'rf_scripts/run_ftfree_eval_model_large_text_only.sh'
    cmds_script_path_model_large_vision_only = 'rf_scripts/run_ftfree_eval_model_large_vision_only.sh'
    cmds_script_path_model_large_vision_and_text = 'rf_scripts/run_ftfree_eval_model_large_vision_and_text.sh' 

    main(datasets_links_filepath, cmds_script_path_model_large_text_only, num_gpus=7, text=True, vision=False, dataset_type=dataset_type)  
    main(datasets_links_filepath, cmds_script_path_model_large_vision_only, num_gpus=8, text=False, vision=True, dataset_type=dataset_type)    
    main(datasets_links_filepath, cmds_script_path_model_large_vision_and_text, num_gpus=8, text=True, vision=True, dataset_type=dataset_type)

        