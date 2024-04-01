# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,                # clip embedding size
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)
        
        # the weights for this layer is saved in the model weights with layer name: roi_heads.box_predictor.*.cls_score.linear.weight   ( each cascade step of roi_box_head has this linear layer)
        self.linear = nn.Linear(input_size, zs_weight_dim)            # zs_weight_dim = 512 so that we can compute similarity with clip embeddings.
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None, new_fedloss_mode=False):
        '''
        Inputs:
            x: B x D'                                       
            classifier_info: (C', C' x D)
        '''

        ## the x in argument is of size 256 x 1024, where 1024 is the size of box features, and 256 are basically the number of proposals(?)
        x_embedding = self.linear(x)             # x is 256 x512 here: compatible with clip embeddings
        # shape of x_embedding is (num_proposals, clip_embd size=512)
        if classifier is not None:
            if new_fedloss_mode:                                        # using new version of fedloss
                zs_weight = classifier[0].permute(1,0).contiguous()
            else:
                zs_weight = classifier.permute(1, 0).contiguous() # D x C'                (C' is less than num_classes if using fedloss)
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x_embedding, p=2, dim=1)
            x = torch.mm(x, zs_weight)                                      # multiplying image embedding x with zs_weight (or clip text embedding)
        else:
            x = torch.mm(x_embedding, zs_weight)
        if new_fedloss_mode:
            cls_inds = classifier[1]#[0]
            num_batches = cls_inds[0].shape[0]
            effective_batch_size = x.shape[0]//num_batches
            gt_negative_mask = cls_inds[0].repeat_interleave(effective_batch_size, dim=0) 
            assert(gt_negative_mask.shape[0]==x.shape[0]), "The math of extracting number of batches from cls_inds doesn't work"
            x = x*gt_negative_mask                 # mask has ones for all FSOD GT+ Negative categories (sampled or extracted from whole GT annotations)


        if self.use_bias:     #false by default
            x = x + self.cls_bias
        return x, x_embedding