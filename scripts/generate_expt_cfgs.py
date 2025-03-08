# import yaml
import ruamel.yaml
import numpy as np  
import os
import glob  
import shutil
import csv
from pathlib import Path
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.utils import get_clean_ann_data, get_rf_cat_info 
import json 


def generate_cat_info_for_dset(dataset_name_in_cfg, link_dataset_name, sv_path):
    clean_annos, _ = get_clean_ann_data(Path(MetadataCatalog.get(dataset_name_in_cfg).image_root) / '_annotations.coco.json')
    categories, cat_img_count_list = get_rf_cat_info(clean_annos)

    with open(sv_path / f'{link_dataset_name}_cat_info.json', 'w') as f:
        json.dump(cat_img_count_list, f)

    return categories, cat_img_count_list, str(sv_path / f'{link_dataset_name}_cat_info.json')


# shots = [5,10,30]
shots = [10]

base_cfg_path = '/home/anishmad/msr_thesis/rf_fsod_baselines/configs/nuimages_cr/code_release_v2/'
sv_cfg_path = Path('/home/anishmad/msr_thesis/rf_fsod_baselines/configs/rf_configs/')
base_cfgs = glob.glob(os.path.join(base_cfg_path,'*.yaml'))
yaml = ruamel.yaml.YAML()

base_repo_path = Path('/home/anishmad/msr_thesis/rf_fsod_baselines/')
datasets_links_filepath = base_repo_path / 'datasets_links.csv'
assert datasets_links_filepath.exists(), f"File not found: {datasets_links_filepath}"
links = []
with open(datasets_links_filepath, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        url = row[0]
        links.append(url)

links = links[1:]
for link in links:
    dataset_name = link.split("/")[-2]    #needs to change with new dataset links file
    updated_dataset_name = dataset_name.replace("-", "_")
    for base_cfg in base_cfgs:
        for shot in shots:
            tgt_dir = sv_cfg_path / dataset_name / f'{shot}_shots'
            tgt_dir.mkdir(parents=True, exist_ok=True)
        
            if "Base" in os.path.basename(base_cfg):           # if it is a Base config , just copy it
                b_cfgname = str(os.path.basename(base_cfg).split('.')[0] + '.yaml')
                new_cfg_sv_path = os.path.join(tgt_dir, b_cfgname)

                shutil.copyfile(base_cfg, new_cfg_sv_path)
                continue
                
            elif not ("zeroshot" in os.path.basename(base_cfg)) :                # only need zeroshot config
                continue

            # for seed in seeds:
            with open(base_cfg) as f:
                # cfg_deets = yaml.safe_load(f)
                cfg_deets = yaml.load(f)

            # import ipdb; ipdb.set_trace()
            
            cur_dset_name = cfg_deets['DATASETS']['TRAIN']
            new_dset_name = f'("{updated_dataset_name}_train_best_split_shots_{shot}",)'
            cfg_deets['DATASETS']['TRAIN'] = new_dset_name

            trainset_length = len(DatasetCatalog.get(f"{updated_dataset_name}_train_best_split_shots_{shot}"))
            valset_length = len(DatasetCatalog.get(f"{updated_dataset_name}_valid_best_split_shots_{shot}"))

            batchsize = min(8, min(trainset_length, valset_length))
            cfg_deets['SOLVER']['IMS_PER_BATCH'] = batchsize
            batchsize_scaling_factor = batchsize/8

            cur_dset_test_name = cfg_deets['DATASETS']['TEST']
            new_dset_test_name = f'("{updated_dataset_name}_test_best_split_shots_{shot}",)'
            cfg_deets['DATASETS']['TEST'] = new_dset_test_name

            cfg_deets['MODEL']['ROI_BOX_HEAD']['ALL_ANN_FILE'] = os.path.join(MetadataCatalog.get(f"{updated_dataset_name}_test_best_split_shots_{shot}").image_root , '_annotations.coco.json')

            sv_path = Path(f'datasets/metadata/cat_info/')
            sv_path.mkdir(parents=True, exist_ok=True)
            
            cats, cat_info_count, cat_count_path = generate_cat_info_for_dset(f"{updated_dataset_name}_test_best_split_shots_{shot}", updated_dataset_name, sv_path)
            cfg_deets['MODEL']['ROI_BOX_HEAD']['CAT_FREQ_PATH'] = cat_count_path

            cfg_deets['DATASETS']['ALL_CLASSES'] = str(cats)
            cfg_deets['DATASETS']['NUM_ORIG_CLASSES'] = int(len(cats))

            cfg_deets['MODEL']['TRAIN_NUM_CLASSES'] = [int(len(cats))]
            cfg_deets['MODEL']['TEST_NUM_CLASSES'] = [int(len(cats))]

            total_iters = int(len(cats))*450  # used 8k iterations for 18 classes in nuimages: using roughly the same ratio here
            val_period = int(total_iters/25)  # 25 validation runs max

            base_lr = cfg_deets['SOLVER']['BASE_LR']
            scaled_base_lr = float(base_lr) * batchsize_scaling_factor   # linear scaling rule
            cfg_deets['SOLVER']['BASE_LR'] = scaled_base_lr
            cfg_deets['SOLVER']['MAX_ITER'] = int(total_iters / batchsize_scaling_factor)   # linear scaling rule
            cfg_deets['TEST']['VAL_PERIOD'] = int(val_period / batchsize_scaling_factor)   # linear scaling rule

            new_cfg_sv_path = os.path.join(tgt_dir, os.path.basename(base_cfg).split('.')[0]+'.yaml')  #TODO: replace shots=5 as well
            # import ipdb; ipdb.set_trace()
            with open(new_cfg_sv_path, "w") as f:
                yaml.dump(cfg_deets, f)
