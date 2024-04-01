# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import numpy as np
from torch.nn import functional as F
import os 
from detectron2.data import MetadataCatalog

def load_class_freq(
    path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=1.0, use_ann_count=False):
    cat_info = json.load(open(path, 'r'))

    if use_ann_count:
        cat_info = torch.tensor(
            [c['instance_count'] for c in sorted(cat_info, key=lambda x: x['id'])]) 
    else:
        cat_info = torch.tensor(
            [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight

def get_fed_loss_inds_deterministic_with_negs(gt_classes, file_names, C, dataset_name=None, img_neg_cat_map=None):
    ### Consider example
    # appeared_mask = [1,1,1,1,1,1]
    # Let categories present in  image be 1 and 4 (either from predictions or True GT)
    # now appeared_mask = [1,0,1,1,0,1]
    # Let GT FSOD = 4 (This is in the given annotation)
    # appeared_mask = [1,0,1,1,1,1]
    # this final mask tells us GT FSOD + {Categories for sure not in image}=Negatives

    assert dataset_name is not None, f"Dataset name is {dataset_name}, fix it"
    
    num_batches = len(gt_classes)
    all_appeared_mask = []
    all_neg_gt_cats = []

    for idx, file_name in enumerate(file_names):
        appeared_mask = torch.zeros(C+1).to(gt_classes[0].device)
        if 'lvis' in dataset_name:
            img_name = os.path.join(file_name.split('/')[-2],os.path.basename(file_name))
        else:
            img_name = os.path.basename(file_name)


        num_gt_classes = gt_classes[idx].shape[0]
        appeared_mask[torch.unique(gt_classes[idx])] = 1
        
        neg_gt_cats = img_neg_cat_map[img_name]         
        if neg_gt_cats:
            appeared_mask[np.array(neg_gt_cats)] = 1
        all_neg_gt_cats.append(torch.Tensor(neg_gt_cats))
        all_appeared_mask.append(appeared_mask.unsqueeze(0).expand(num_gt_classes, -1))
    all_appeared_cats = torch.unique(torch.cat(all_neg_gt_cats))

    return torch.cat(all_appeared_mask, dim=0), all_appeared_cats

def get_fed_loss_inds_deterministic2(gt_classes, img_to_cat_map, file_names, C, dataset_name=None):

    ### Consider example
    # appeared_mask = [1,1,1,1,1,1]
    # Let categories present in  image be 1 and 4 (either from predictions or True GT)
    # now appeared_mask = [1,0,1,1,0,1]
    # Let GT FSOD = 4 (This is in the given annotation)
    # appeared_mask = [1,0,1,1,1,1]
    # this final mask tells us GT FSOD + {Categories for sure not in image}=Negatives

    assert dataset_name is not None, f"Dataset name is {dataset_name}, fix it"
    
    num_batches = len(gt_classes)
    all_appeared_mask = []
    all_neg_gt_cats = []

    for idx, file_name in enumerate(file_names):
        appeared_mask = torch.ones(C+1).to(gt_classes[0].device)
        if 'lvis' in dataset_name:
            img_name = os.path.join(file_name.split('/')[-2],os.path.basename(file_name))
        else:
            img_name = os.path.basename(file_name)

        cats_in_img = img_to_cat_map[img_name]

        appeared_mask[np.unique(cats_in_img)] = 0          # put all true GT categories to 0 except FSOD GT(this means there is a 1 for all true negative categories + GT FSOD annotation)
        num_gt_classes = gt_classes[idx].shape[0]
        appeared_mask[torch.unique(gt_classes[idx])] = 1
        
        neg_gt_cats = torch.where(appeared_mask==1)[0]            # leave out bg class
        all_neg_gt_cats.append(neg_gt_cats)
        all_appeared_mask.append(appeared_mask.unsqueeze(0).expand(num_gt_classes, -1))
    all_appeared_cats = torch.unique(torch.cat(all_neg_gt_cats))

    return torch.cat(all_appeared_mask, dim=0), all_appeared_cats

def get_fed_loss_inds_prob(gt_classes, num_sample_cats, C, weight=None, inverse_weights=False):

    all_appeared_mask = []
    for idx, gt_cls in enumerate(gt_classes):
        num_gt_classes = gt_cls.shape[0]
        appeared_mask = torch.zeros(C+1).to(gt_classes[0].device)
        appeared = torch.unique(gt_cls)
        prob = appeared.new_ones(C + 1).float()
        prob[-1] = 0
        if len(appeared) < num_sample_cats:
            if weight is not None:
                if inverse_weights:
                    weight=1.0/weight
                prob[:C] = weight.float().clone()
            prob[appeared] = 0

            more_appeared = torch.multinomial(                      # sampled indices corresponding to classes not appearing in GT proportional to the weight provided(based on freq of classes)
                    prob, num_sample_cats - len(appeared),
                    replacement=False)
            appeared = torch.cat([appeared, more_appeared])         # appeared now consists of sampled negative classes along with gt classes

        appeared_mask[appeared] = 1
        all_appeared_mask.append(appeared_mask.unsqueeze(0).expand(num_gt_classes, -1))
    return torch.cat(all_appeared_mask, dim=0), None
        


def get_fed_loss_inds_deterministic(gt_classes, img_to_cat_map, file_names, C):

    num_batches = len(gt_classes)

    all_neg_cats = []
    appeared_mask = torch.ones((num_batches, C+1)).to(gt_classes[0].device)
    for idx, file_name in enumerate(file_names):
        all_cats = torch.arange(C).to(gt_classes[0].device)
        img_name = os.path.basename(file_name)
        cats_in_img = img_to_cat_map[img_name]
        appeared_mask[idx][np.unique(cats_in_img)] = 0          # put all true GT categories to 0 except FSOD GT(this means there is a 1 for all true negative categories + GT FSOD annotation)
        appeared_mask[idx][gt_classes[idx]] = 1
        neg_cats = torch.where(appeared_mask[idx]==1)[0]            # leave out bg class
        all_neg_cats.append(neg_cats)
    all_appeared_cats = torch.unique(torch.cat(all_neg_cats))
    return appeared_mask.long(), all_appeared_cats
        


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None, keep_inds=None, inverse_weights=False):
    # gt classes based on target boxes, 
    # num_sample_cats = 50 or whatever is specified as the subset for dynamic classifier
    # C : total number of classes in LVIS
    # weight: weight corresponding to frequency of classes in LVIS.

    
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0 
    if len(appeared) < num_sample_cats:
        if weight is not None:
            if inverse_weights:
                weight=1.0/weight
            prob[:C] = weight.float().clone()
        prob[appeared] = 0

        if keep_inds is not None:
            keep_inds_cls_mask = torch.ones(C+1).long().to(gt_classes.device)
            keep_inds_cls_mask[keep_inds] = 0                 # all don't care classes or the classes we don't want to include in the set of negatives are 1 here.

            prob[keep_inds_cls_mask==1] = 0                    # only weights of classes which occur as  specified (negatives - appeared)

            num_to_sample = min(torch.count_nonzero(prob).item(), num_sample_cats - len(appeared))
            more_appeared = torch.multinomial(                      # sampled indices corresponding to classes not appearing in GT proportional to the weight provided(based on freq of classes)
                prob, num_to_sample,
                replacement=False)
        else:
            more_appeared = torch.multinomial(                      # sampled indices corresponding to classes not appearing in GT proportional to the weight provided(based on freq of classes)
                prob, num_sample_cats - len(appeared),
                replacement=False)
        appeared = torch.cat([appeared, more_appeared])         # appeared now consists of sampled negative classes along with gt classes
    return appeared


def reset_cls_test(model, cls_path, num_classes):
    model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        zs_weight = torch.tensor(                                  # 512 x num classes
            np.load(cls_path), 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    elif isinstance(cls_path, np.ndarray):
        zs_weight = torch.tensor(
            cls_path, 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
        dim=1) # D x (C + 1)
    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    for k in range(len(model.roi_heads.box_predictor)):
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
