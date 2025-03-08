# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
from pathlib import Path

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer#, PeriodicCheckpointer
from detectron2.utils.checkpointing import PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    inference_on_dataset_custom,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader, build_detection_val_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler
from tools.get_clip_features import compute_clip_features

import json
import torch
import numpy as np
import sys
import clip
from collections import Counter

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from detic.custom_solver import build_custom_optimizer
from detic.evaluation.oideval import OIDEvaluator
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator

from detic.modeling.utils import reset_cls_test#, reset_cls_train

logger = logging.getLogger("detectron2")

def do_test(cfg, model, path_suffix=None):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        custom_vocab_bool = len(cfg.DATASETS.ALL_CLASSES)>cfg.MODEL.TEST_NUM_CLASSES[d]
        if cfg.MODEL.RESET_CLS_TESTS:
                reset_cls_test(
                    model,
                    compute_clip_features(MetadataCatalog.get(dataset_name).json_file)[0],
                    # cfg.MODEL.TEST_CLASSIFIERS[d],
                    cfg.MODEL.TEST_NUM_CLASSES[d])
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)         # mapper is assigned if it  is  None, just follow path trace of this function to _test_loader_from_config() in build.py
        if path_suffix is not None:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, path_suffix, "inference_{}".format(dataset_name))
        else:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = inference_on_dataset_custom(
            model, data_loader, evaluator, cfg)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_validation(cfg, data_loader, model, writers=None, train_iteration=-1):
    start_iter = 0
    print('Running Validation')
    with EventStorage(train_iteration) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        total_loss = 0
        total_loss_dict = None
        data_time = data_timer.seconds()
        # import ipdb; ipdb.set_trace()
        for iteration, data in enumerate(data_loader):
            print(f'Running val iteration {iteration}')

            loss_dict = model(data, valmode=True)
            loss_dict = {'val_'+str(k):v for k, v in loss_dict.items()}

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            
            if total_loss_dict is None:
                total_loss_dict = loss_dict_reduced
            else:
                total_loss_dict = dict(Counter(total_loss_dict) + Counter(loss_dict_reduced))
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            total_loss += losses_reduced

        avg_total_loss = total_loss / (iteration+1)
        avg_total_loss_dict = { k: v/(iteration+1) for k, v in total_loss_dict.items()}

        storage.put_scalars(val_data_time=data_time)

        if comm.is_main_process():
            storage.put_scalars(
                val_total_loss=avg_total_loss, **avg_total_loss_dict)

        for writer in writers:
            writer.write()

        data_timer.reset()

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))
        
        return avg_total_loss

def do_train(cfg, model, resume=False):
    
    model.train()
    
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    periodic_checkpointer_best_ckpt = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, file_prefix = "best_model", best_model_mode = True
    )
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    val_writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "val_metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    use_custom_mapper = cfg.WITH_IMAGE_LABELS
    MapperClass = CustomDatasetMapper if use_custom_mapper else DatasetMapper
    mapper = MapperClass(cfg, True, augmentations=[]) if cfg.INPUT.CUSTOM_AUG == '' else \
        DetrDatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == 'DETR' else \
        MapperClass(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper)

    if cfg.FP16:
        scaler = GradScaler()

    if cfg.TEST.VAL_PERIOD > 0:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
        else DatasetMapper(                                                  # check val_loader_from_config() to edit augmentations in Mapper. The issue is that when mapper is None, augmentations were applied by default. Fixed those now with empty lists.
            cfg, False, augmentations=build_custom_augmentation(cfg, False))
        if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler', 'InferenceSampler']:
            val_data_loader = build_detection_val_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        total_val_loss_dict = None
        best_val_loss = 99999
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data, valmode=False)
            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (cfg.TEST.VAL_PERIOD > 0
                and iteration % cfg.TEST.VAL_PERIOD == 0):
                cur_val_loss = do_validation(cfg, val_data_loader, model, val_writers, train_iteration=iteration)
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_iteration = iteration
                    periodic_checkpointer_best_ckpt.step(iteration)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):# and not iteration % cfg.TEST.VAL_PERIOD != 0:
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))
        
    if cfg.TEST.VAL_PERIOD > 0:
        old_best_model_name  = os.path.join(cfg.OUTPUT_DIR, "best_model_final.pth")
        new_best_model_name = os.path.join(cfg.OUTPUT_DIR, f"best_model_final_{best_iteration}.pth")
        os.rename(old_best_model_name, new_best_model_name)
        return new_best_model_name
    else:
        return None

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.OUTPUT_DIR_PREFIX is not None:
        suffix_file_name = args.config_file.split('configs/rf_configs/')[1].split('.yaml')[0]
        # suffix_file_name = args.config_file.replace('configs/rf_configs/','')[:-5]
        output_dir = Path(cfg.OUTPUT_DIR_PREFIX) / suffix_file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg.OUTPUT_DIR = str(output_dir)

    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg


def main(args):

    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    
    else:
        if cfg.MODEL.RESET_CLS_TRAIN:
            dataset_name = cfg.DATASETS.TRAIN[0]
            reset_cls_test(
                model,
                compute_clip_features(MetadataCatalog.get(dataset_name).json_file)[0],
                # cfg.MODEL.TRAIN_CLASSIFIERS[0],
                cfg.MODEL.TRAIN_NUM_CLASSES[0])
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    best_model_path = do_train(cfg, model, resume=args.resume)

    if best_model_path is not None: # i.e using valset 
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                best_model_path, resume=args.resume
            )
        print('Best model Path', best_model_path)
        print('Running Test script with best model weights')
        return do_test(cfg, model, path_suffix=best_model_path.split('.pth')[0])
    else:
        return []


if __name__ == "__main__":
    args = default_argument_parser()
    
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
