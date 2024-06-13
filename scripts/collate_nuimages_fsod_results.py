import csv
import os
import numpy as np 
import glob

orig_base_dir = '/home/anishmad/msr_thesis/mqdet/results/nuimages_fsod/'
shots = [5,10,30]

base_expt_suffix_names = ['tiny_text_only', 'tiny_vision_only', 'tiny_text_and_vision', 'large_text_only', 'large_vision_only', 'large_text_and_vision']

expt_names = ['GLIP-T-Text', 'GLIP-T-Vision', 'GLIP-T-Text+Vision', 'GLIP-L-Text', 'GLIP-L-Vision', 'GLIP-L-Text+Vision']

seeds = [0,1,2]

header_row = ['expt type', 'metric', 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'emergency', 'motorcycle', 'bicycle', 'adult', 'child', 'police_officer', 'construction_worker', 'personal_mobility', 'stroller', 'pushable_pullable', 'barrier', 'traffic_cone', 'debris', 'avg']

data_to_write = []

for shot in shots:
    for seed in seeds:
        base_dir = os.path.join(orig_base_dir, f'{shot}_shots_seed_{seed}')
        data_to_write.append([f'Seed {seed}'])
        data_to_write.append(header_row)
        for idx, base_expt_suffix in enumerate(base_expt_suffix_names):
            # cur_cfg_name = f'{cfg}_shots{5}_seed{seed}_predconf_20pc'          # use for all non self training expts (since there is a typo: for all shots, the cfg name has "5_shots" in it)
            # cur_cfg_name = f'{cfg}_shots{5}_seed_{seed}'         # use for all non self training expts (since there is a typo: for all shots, the cfg name has "5_shots" in it)
            # cur_cfg_name = f'{cfg}_shots{num_shots}_seed{seed}_predconf_20pc'
            
            result_path = os.path.join(base_dir, f'model_{base_expt_suffix}')
            # import ipdb; ipdb.set_trace()
            result_csv_paths = glob.glob(f'{result_path}*/**/*.csv', recursive=True)
            # import ipdb; ipdb.set_trace()
            if len(result_csv_paths)==0:
                continue
            assert len(result_csv_paths)==1, f"Check logic for loading csv, cur csv paths are {result_csv_paths}, {result_path}"
            
            with open(result_csv_paths[0]) as f:
                dd = csv.reader(f, delimiter=',')
                data = list(dd)
                new_data = [expt_names[idx]]
                new_data.extend(data[1])
                data_to_write.append(new_data)

    data_to_write.append([])


collated_res_path = os.path.join(orig_base_dir, f'collated_results.csv')

with open(collated_res_path, 'w') as f:
    writer = csv.writer(f)
    for datarow in data_to_write:
        if datarow == []:
            writer.writerow(['\n'])
        writer.writerow(datarow)