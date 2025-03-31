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

def generate_cat_info_for_dset(dataset_name, shots):
    # clean_annos, _ = get_clean_ann_data(Path(MetadataCatalog.get(dataset_name_in_cfg).image_root) / '_annotations.coco.json')
    # categories, cat_img_count_list = get_rf_cat_info(clean_annos)
        
    # with open(sv_path / f'{link_dataset_name}_cat_info.json', 'w') as f:
    #     json.dump(cat_img_count_list, f)

    # return categories, cat_img_count_list, str(sv_path / f'{link_dataset_name}_cat_info.json')
    data_path = f'gen_data/{dataset_name}/fsod_data_detectron/train/{dataset_name}_fsod_train_best_split_shots_{shots}.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    num_classes = len(categories)

    return num_classes 

# shots = [5,10,30]
shots = [10]

base_cfg_path = 'configs/base_rf_config/base_rf_10_shots_fsod.yaml'
sv_cfg_path = Path('configs/rf_configs')
sv_cfg_path.mkdir(exist_ok=True, parents=True)

# base_cfgs = glob.glob(os.path.join(base_cfg_path,'*.yaml'))
yaml = ruamel.yaml.YAML()

datasets_links_filepath = 'datasets_links.csv'
assert os.path.exists(datasets_links_filepath), f"File not found: {datasets_links_filepath}"
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
    # for base_cfg in base_cfgs:
    for shot in shots:
        tgt_dir = sv_cfg_path / dataset_name / f'{shot}_shots'
        tgt_dir.mkdir(parents=True, exist_ok=True)
    
        # for seed in seeds:
        with open(base_cfg_path) as f:
            # cfg_deets = yaml.safe_load(f)
            cfg_deets = yaml.load(f)

        # import ipdb; ipdb.set_trace()
        num_classes = generate_cat_info_for_dset(dataset_name, shot)
        cfg_deets['MODEL']['ATSS']['NUM_CLASSES'] = num_classes
        cfg_deets['MODEL']['FCOS']['NUM_CLASSES'] = num_classes
        cfg_deets['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES'] = num_classes
        cfg_deets['MODEL']['DYHEAD']['NUM_CLASSES'] = num_classes


        train_register = f'{updated_dataset_name}_train'
        val_register = f'{updated_dataset_name}_val'
        test_register = f'{updated_dataset_name}_test'
        
        cfg_deets['DATASETS']['REGISTER'] = {}
        for mode in ['train', 'val', 'test']:
            cfg_deets['DATASETS']['REGISTER'][f'{updated_dataset_name}_{mode}'] = {}  
            cfg_deets['DATASETS']['REGISTER'][f'{updated_dataset_name}_{mode}']['img_dir'] = f'data/{dataset_name}/{mode}'
            cfg_deets['DATASETS']['REGISTER'][f'{updated_dataset_name}_{mode}']['ann_file'] = f'gen_data/{dataset_name}/fsod_data_detectron/{mode}/{dataset_name}_fsod_{mode}_best_split_shots_{shot}.json'

        # cur_dset_name = cfg_deets['DATASETS']['TRAIN']
        new_dset_name = f'("{train_register}",)'
        cfg_deets['DATASETS']['TRAIN'] = new_dset_name
        new_test_dset_name = f'("{test_register}",)'
        cfg_deets['DATASETS']['TEST'] = new_test_dset_name

        cfg_deets['DATASETS']['FEW_SHOT'] = shot

        cfg_deets['INPUT']['MIN_SIZE_TRAIN'] = 640
        cfg_deets['INPUT']['MAX_SIZE_TRAIN'] = 640
        cfg_deets['INPUT']['MIN_SIZE_TEST'] = 640
        cfg_deets['INPUT']['MAX_SIZE_TEST'] = 640

        new_cfg_sv_path = os.path.join(tgt_dir, 'multimodal_prompting.yaml')  #TODO: replace shots=5 as well
        # import ipdb; ipdb.set_trace()
        with open(new_cfg_sv_path, "w") as f:
            yaml.dump(cfg_deets, f)
