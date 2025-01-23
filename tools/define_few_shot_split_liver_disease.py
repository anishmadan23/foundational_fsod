
import argparse
import json
import os
import random
import numpy as np 
from collections import defaultdict
from detectron2.data.datasets.coco import load_coco_json, convert_to_coco_dict_with_dset
from copy import deepcopy
import glob


argparser = argparse.ArgumentParser()
argparser.add_argument('--data_train_path', type=str, default='/data3/anishmad/roboflow_data/liver_disease/train/_annotations.coco.json')
argparser.add_argument('--data_val_path', type=str, default='/data3/anishmad/roboflow_data/liver_disease/valid/_annotations.coco.json')
argparser.add_argument('--base_save_path', type=str, default='/home/anishmad/msr_thesis/detic-lt3d/data2/')
argparser.add_argument('--dset_name', type=str, default='liver_disease')
argparser.add_argument('--num_splits', type=int, default=1)
argparser.add_argument('--num_shots', type=int, default=10)

args= argparser.parse_args()


def get_save_path_seeds(cls, shots, seed, base_save_path, dataset='nuimages', suffix_save_path='', valmode=False):

    filename_suffix = 'trainval'
    prefix = "full_box_{}shot_{}_{}".format(shots, cls, filename_suffix)
    if valmode:
        save_dir = os.path.join(base_save_path , "datasets", dataset, suffix_save_path, "val", "seed" + str(seed))
    else:
        save_dir = os.path.join(base_save_path , "datasets", dataset, suffix_save_path, "seed" + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path

def generate_seed_split(data, seed, valmode=False, dset_name='nuimages', suffix_save_path=''):
    # data_path = "/home/anishmad/msr_thesis/glip/DATASET/nuimages/annotations/nuimages_v1.0-train.json"
    # data = json.load(open(data_path))
    
    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat)

    id2img = {}
    for i in data["images"]:
        id2img[i["id"]] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for c in ID2CLASS.keys():

            
        print('class',c)
        img_ids = defaultdict(list)

        for a in anno[c]:
            # if a["image_id"] in img_ids:
            img_ids[a["image_id"]].append(a)

                    
        sample_shots = []
        sample_imgs = []
        if valmode:
            all_shots = [10]
        else:
            all_shots = [10]
    
        for orig_shots in all_shots:
            print('orig_shots', orig_shots)
            shots=orig_shots
            while True:
                print("num imgs vs shots", len(list(img_ids.keys())), shots)
                if len(list(img_ids.keys()))<shots:
                    print(shots, c)                               # if printed, it means images for that specific class in the split are less than the shots
                    
                    shots=len(list(img_ids.keys()))
                    imgs = list(img_ids.keys())
                else:
                    imgs = random.sample(list(img_ids.keys()), shots)      

                for img in imgs:
                    skip = False
                    for s in sample_shots:
                        if img == s["image_id"]:
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_ids[img]) + len(sample_shots) > shots:
                        continue
                       
                    sample_shots.extend(img_ids[img])
                    sample_imgs.append(id2img[img])
                    if len(sample_shots) == shots:
                        break
                if len(sample_shots) == shots or (len(sample_shots) <=shots and valmode):
                    break
            new_data = {
                # "info": data["info"],
                # "licenses": data["licenses"],
                "images": sample_imgs,
                "annotations": sample_shots,
            }
            save_path = get_save_path_seeds(
                ID2CLASS[c], orig_shots, seed, args.base_save_path, dataset=dset_name, suffix_save_path=suffix_save_path, valmode=valmode
            )
            
            new_data["categories"] = new_all_cats
            with open(save_path, "w") as f:
                json.dump(new_data, f)


def modify_dataset_for_few_shot(dataset, base_split_path, shots=5, seed=1, valmode=False):

    if valmode:
        all_split_files = glob.glob(os.path.join(base_split_path, 'val', 'seed'+str(seed), f'full_box_{shots}shot_*.json'))
    else:
        all_split_files = glob.glob(os.path.join(base_split_path, 'seed'+str(seed), f'full_box_{shots}shot_*.json'))
        
    new_dataset_dict = {}
    for cls_split_file in all_split_files:
        # get matching info from dataset and add it to a new list which would work as new FS dataset
        cls_info = json.load(open(cls_split_file, 'r'))
        for idx_ann, cls_info_ann in enumerate(cls_info['annotations']):
            img_id = cls_info_ann['image_id']
            all_data_info = dataset[img_id]                              # corresponding info in all data 
            assert(img_id==all_data_info['image_id']), f"Image id of split file is {img_id} and its corresponding one in all dataset is {all_data_info['image_id']}"
            match_flag=0
            for all_data_info_ann in all_data_info['annotations']:
                if cls_info_ann['bbox'] == all_data_info_ann['bbox']:
                    match_flag=1
                    if img_id in new_dataset_dict:
                        new_dataset_dict[img_id]['annotations'].append(all_data_info_ann)
                    else:
                        new_dataset_dict[img_id] = deepcopy(all_data_info)
                        new_dataset_dict[img_id]['annotations'] = [all_data_info_ann]          # replace all image annotations by the split annotation only.                    break
            # if match_flag==0:
            assert(match_flag==1), "check annotation as bbox match was not found"

    new_dataset = list(new_dataset_dict.values())
    
    return new_dataset


if __name__ == '__main__':
    
    data_train = json.load(open(args.data_train_path))

    ID2CLASS = {}
    for cat_info in data_train['categories']:
        ID2CLASS[cat_info['id']] = cat_info['name']
    

    CLASS2ID = {v: k for k, v in ID2CLASS.items()}


    ann_by_cls_imgid = defaultdict(list)
    for idx, ann_info in enumerate(data_train['annotations']):
        ann_by_cls_imgid[ann_info['image_id']].append(ann_info['category_id'])

    print("Num images in split", len(data_train['images']))


    random_split_seeds = np.arange(args.num_splits)
    num_img_ids = len(data_train['images'])

    for random_split_seed in random_split_seeds:
        random.seed(random_split_seed)
        np.random.seed(random_split_seed)
        # sel_img_ids = random.sample(list(np.arange(num_img_ids)), k=int(p_valset_size*num_img_ids))
        tr_seed_data = []
        print(f'Converting to annotation fmt for seed {random_split_seed}')
        generate_seed_split(data_train, random_split_seed, valmode=False, dset_name=args.dset_name, suffix_save_path='fsod_data')


    ### Few Shot Setup (Converting to JSON format to register it in Detectron2)
    # IMP: Add dataset path related info in detectron along with registering it. (https://github.com/anishmadan23/detectron2-ffsod/blob/main/detectron2/data/datasets/builtin.py)

    dset = load_coco_json(args.data_train_path, os.path.dirname(args.data_train_path), dataset_name='liver_disease_all_cls_train')
    dset_dict_by_imgid = {x['image_id']:x for x in dset}          # needed for coco as image id aren't [0,... n-1] like in nuimages.

    all_shots = [10]
    # all_seeds = np.arange(10)
    all_seeds = [0]
    base_fs_split_path = os.path.join(args.base_save_path, 'datasets', args.dset_name, 'fsod_data')    #saved individual splits here
    save_new_anno_path = os.path.join(args.base_save_path, 'datasets', args.dset_name, 'fsod_data_detectron')
    os.makedirs(save_new_anno_path, exist_ok=True)

                    
    # #### Method to filter complete dataset acc to few shot split files
    # - Start by loading full dataset. We then load FS split files.
    # - For each annotation (A) in split file, we match it to some image's (I') annotation (A') in the full dataset, and use that full dataset image info , by modifying annotation according to FS split annotation. (I': A -> A')

    for seed in all_seeds:
        for shots in all_shots:
            few_shot_dset = modify_dataset_for_few_shot(dset_dict_by_imgid, base_fs_split_path, shots=shots, seed=seed, valmode=False)
            few_shot_dset_json = convert_to_coco_dict_with_dset('liver_disease_all_cls_train', few_shot_dset)
            
            # few_shot_dset_val = modify_dataset_for_few_shot(dset_dict_by_imgid, base_fs_split_path, shots=shots, seed=seed, valmode=True)
            # few_shot_dset_val_json = convert_to_coco_dict_with_dset('nuimages_all_cls_train_no_wc', few_shot_dset_val)
            
            train_fs_filename = f'{args.dset_name}_fsod_train_seed_{seed}_shots_{shots}.json'
            val_fs_filename = f'{args.dset_name}_fsod_val_seed_{seed}_shots_{shots}.json'
            
            with open(os.path.join(save_new_anno_path, train_fs_filename), 'w') as f:
                json.dump(few_shot_dset_json, f)
