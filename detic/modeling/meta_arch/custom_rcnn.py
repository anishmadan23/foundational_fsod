# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from collections import defaultdict
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm
import os
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.data.datasets.coco import load_coco_json

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds, get_fed_loss_inds_deterministic

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        modify_neg_loss = False,
        use_zs_preds_nl = False,
        zs_preds_path_nl = None,
        zs_conf_thresh = None,
        use_gt_nl = False,
        gt_path_nl = None,

        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        self.modify_neg_loss = modify_neg_loss
        self.use_zs_preds_nl = use_zs_preds_nl
        self.zs_preds_path_nl = zs_preds_path_nl 
        self.use_gt_nl = use_gt_nl 
        self.gt_path_nl = gt_path_nl
        self.zs_conf_thresh = zs_conf_thresh

        if modify_neg_loss and use_gt_nl:
            self.gt_annos = self.get_anno_from_gt_file()
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
            self.keep_neg_cls_inds = kwargs.pop('keep_neg_cls_inds')
            self.inverse_weights = kwargs.pop('fed_inverse_weight')
            self.deterministic_fed_loss = kwargs.pop('deterministic_fed_loss')
            self.all_ann_file = kwargs.pop('all_ann_file')

            if self.deterministic_fed_loss:
                all_train_data = load_coco_json(self.all_ann_file, '', dataset_name='_')
        
                self.img_cat_map = {}
                for idx, img_info in enumerate(all_train_data):
                    # img_id = img_info['image_id']
                    
                    all_cats = [x['category_id'] for x in img_info['annotations']]
                    img_name = os.path.basename(img_info['file_name'])
                    
                    self.img_cat_map[img_name] = all_cats


        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

    def get_anno_from_gt_file(self):
        assert self.gt_path_nl is not None, "self.gt_path_nl is None, add correct path"

        with open(self.gt_path_nl, 'r') as f:
            gt_annos = json.load(f)
        
        img_anno_map = defaultdict(list)
        for anno in gt_annos['annotations']:
            img_id = anno['image_id']
            file_name = gt_annos['images'][img_id]['file_name']
            img_anno_map[file_name].append(anno)

        return img_anno_map

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'modify_neg_loss': cfg.MODEL.NEG_LOSS.MODIFY,
            'use_zs_preds_nl': cfg.MODEL.NEG_LOSS.USE_ZS_PREDS,
            'zs_preds_path_nl': cfg.MODEL.NEG_LOSS.ZS_PREDS_PATH,
            'use_gt_nl': cfg.MODEL.NEG_LOSS.USE_GT,
            'gt_path_nl': cfg.MODEL.NEG_LOSS.GT_PATH,
            'zs_conf_thresh': cfg.MODEL.NEG_LOSS.ZS_CONF_THRESH,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
                )
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
            ret['keep_neg_cls_inds'] = cfg.MODEL.ROI_BOX_HEAD.KEEP_FED_NEG_CLS_INDS
            ret['fed_inverse_weight'] = cfg.MODEL.ROI_BOX_HEAD.INVERSE_WEIGHTS
            ret['all_ann_file'] = cfg.MODEL.ROI_BOX_HEAD.ALL_ANN_FILE
            ret['deterministic_fed_loss'] = cfg.MODEL.ROI_BOX_HEAD.DETERMINISTIC_FED_LOSS

        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None
        images = self.preprocess_image(batched_inputs)
        if 'file_name' in batched_inputs[0]:
            file_names = [x['file_name'] for x in batched_inputs]
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            features = self.backbone(images.tensor)
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            results, _ = self.roi_heads(images, features, proposals, gt_instances, file_names=file_names)   # call to Cascade ROI class
        else:
            file_names=None
            features = self.backbone(images.tensor)
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, None)   # call to Cascade ROI class

        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], valmode=False):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        
        ann_type = 'box'
        file_names = [x['file_name'] for x in batched_inputs]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None
        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':

            ###### when using _sample_cls_inds()

            # cls_inds = self._sample_cls_inds(gt_instances, ann_type)
            # ind_with_bg = cls_inds[0].tolist() + [-1]               # [gt_inds with -1 for background]
            # cls_features = self.roi_heads.box_predictor[                                  # clip embedding used here (I think)
            #     0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()                    #shape  512x len(ind_with_bg)

            cls_inds = self._sample_cls_inds2(gt_instances, ann_type, file_names=file_names) # inds, inv_inds     # gt_instances here correspond to FSOD annotations or whatever GT given in Instances class format
            cls_features = self.roi_heads.box_predictor[                                  # clip embedding used here (I think)
                0].cls_score.zs_weight.permute(1, 0).contiguous()    
        
        classifier_info = cls_features, cls_inds, caption_features                            #  this classifier _info is used in DeticFastRCNNOutputLayer forward method
        if self.modify_neg_loss:
            if self.use_zs_preds_nl:
                extra_boxes_path = self.zs_preds_path_nl
            elif self.use_gt_nl:
                extra_boxes_path = self.gt_annos
            else:
                raise Exception('Either ZS predictions or GT labels to modify neg loss need to be True')
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances, negloss_boxes_path=extra_boxes_path, file_names=file_names, zs_conf_thresh=self.zs_conf_thresh, zs_negboxes=self.use_zs_preds_nl)
        else:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances, file_names=file_names)    # add pseudotargets here
            
        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:    # default is DeticCascadeROIHeads
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances, file_names, valmode=valmode)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info, file_names=file_names, valmode=valmode)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses


    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight, keep_inds=self.keep_neg_cls_inds, inverse_weights=self.inverse_weights)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))           # new tensor of size (self.num_classes+1,) with all values = len(inds) i.e len(GT+ sampled negative categories)
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)      # for len(inds)=4 , the cls_id_map would look like [4,4,2,4,3,4,4,0,1,4], where 0,1,2,3 are the inds correpsonding to GT+sampled negative classes.
        return inds, cls_id_map


    def _sample_cls_inds2(self, gt_instances, ann_type='box', file_names=None):
        # return boolean mask, num_batches x num_classes to indicate sampled (or deterministic (pos+neg) classes)

        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            batchwise_gt_classes = [x.gt_classes for x in gt_instances]
            gt_classes = torch.cat(batchwise_gt_classes)
            C = len(self.freq_weight)
            num_batches = len(batchwise_gt_classes)
            sampled_mask = torch.zeros((num_batches, C))
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)

        inds_mask, appeared_inds = get_fed_loss_inds_deterministic(batchwise_gt_classes, self.img_cat_map, file_names=file_names, C=C)
        cls_id_map = gt_classes.new_full((self.num_classes + 1,), len(appeared_inds)) 
        cls_id_map[appeared_inds] = torch.arange(len(appeared_inds), device=cls_id_map.device) 

        return inds_mask, cls_id_map
