import json
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_ann_file", default='/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/nuimages_v1.0-train.json', help='Path to training annotation file')
parser.add_argument("--base_save_path", default='/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/no_wc/', help='base save path for modified annotations')
parser.add_argument("--wc_id", type=int, default=13, help='(Wheelchair) Class id for removing its annotations')
args = parser.parse_args()


def get_updated_annotations(orig_annos, cls_id_to_remove=13):
    new_annotations = {}
    new_annotations['images'] = orig_annos['images']
    new_annotations['categories'] = []
    new_annotations['annotations'] = []

    for ann_info in orig_annos['annotations']:
        if ann_info['category_id'] == cls_id_to_remove:      # remove annotation for cls_id_to_remove
            continue
        elif ann_info['category_id'] > cls_id_to_remove:
            new_ann_info = deepcopy(ann_info)
            new_ann_info['category_id'] -= 1 
            new_annotations['annotations'].append(new_ann_info)
        else:
            new_annotations['annotations'].append(ann_info)
            
    for cat_info in orig_annos['categories']:
        if cat_info['id'] == cls_id_to_remove:              # remove entry for cls_id_to_remove from all categories.
            continue
        elif cat_info['id']>cls_id_to_remove:
            new_cat_info = deepcopy(cat_info)
            new_cat_info['id']-=1
            new_annotations['categories'].append(new_cat_info)
        else:
            new_annotations['categories'].append(cat_info)   
    return new_annotations


train_ann_file = args.train_ann_file
val_ann_file = train_ann_file.replace('nuimages_v1.0-train.json', 'nuimages_v1.0-val.json')

os.makedirs(args.base_save_dir, exist_ok=True)

for base_ann_file in [train_ann_file, val_ann_file]:#, train_dummy_ann_file, val_dummy_ann_file]:
    with open(base_ann_file, 'rb') as f:
        dd = json.load(f)
    updated_annos = get_updated_annotations(dd, args.wc_id)

    new_save_path = os.path.join(args.base_save_dir, os.path.basename(base_ann_file))
    with open(new_save_path, 'w') as f:
        json.dump(updated_annos, f)