import os
import numpy as np
import csv

def main(datasets_links_filepath, cmds_script_path):
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
            command = f"""python train_net.py --num-gpus 1 --config-file configs/rf_configs/{dataset}/10_shots/zeroshot.yaml --pred_all_class --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth OUTPUT_DIR_PREFIX /data3/anishmad/msr_thesis/rf_fsod_baselines/evalai_baselines/ \n"""
            # Write the command to the bash script
            bash_script.write(command)




if __name__=='__main__':
    datasets_links_filepath = 'datasets_links.csv'
    cmds_script_path = 'scripts/zero_shot_expts.sh'
    main(datasets_links_filepath, cmds_script_path)
    

        