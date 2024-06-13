# Foundational FSOD - Adapted MQDet Codebase

SWITCH TO MAIN BRANCH FOR DETIC CODEBASE

## Installation and Setup
See [MQDet README](MQDET_README.md) for details on installation and setup.

## DATA Setup (nuImages)

We follow the basic data organization guidelines provided in the [MQDET codebase](https://github.com/YifanXu74/MQ-Det/blob/main/DATA.md) We describe how to setup [nuImages](https://nuscenes.org/nuimages) below. 


## nuImages
1. First, download the nuImages dataset and place/soft-link it in `ROOT/DATASET/`. We provide COCO-style [annotation files here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/nuimages_mqdet_annotation_data/no_wc) for ease of use with this repo.  To create these annotation files from scratch, please refer to the nuImages mmdetection3d data creation script (specifically by running [this file](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/nuimage_converter.py); follow the instructions [here](https://mmdetection3d.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-conversion)) Also note, that MQDet expects annotation indices to start from 1 and not 0, therefore our annotation files take care of this offset as well.
```
$REPOSITORY_ROOT/DATASET
    nuimages/
        images/
            samples/
        <ann_files_downloaded_from_huggingface>
```
2. We provide the [few-shot splits here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/mqdet_data_splits/nuimages). Place them in `nuimages` as well.

3. Finally, register the few shot dataset according to the relative path in the config file. For example, [configs/vision_query_5shot/nuimages/nuimages_5_shots_seed0_fsod.yaml](configs/vision_query_5shot/nuimages/nuimages_5_shots_seed0_fsod.yaml). You should be good to go and run some expts!

## Sample Commands for Foundational FSOD on nuImages


### Step 1: Extract Vision Queries
```bash
python tools/extract_vision_query.py --config_file configs/pretrain/my_configs/mq-glip-l-nuim.yaml --add_config_file configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml --dataset nuim --num_vision_queries 10 --save_path MODEL/nuimages_fsod_10_shots_seed_0/ --add_name large
```
### Step 2: Run FT-Free Evaluation

#### FT Free Evaluation (Vision+Text)
```bash
python -m torch.distributed.launch --nproc_per_node 8 tools/test_grounding_net.py \ 
--config-file configs/pretrain/my_configs/mq-glip-l-nuim.yaml \
--additional_model_config configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml \
VISION_QUERY.QUERY_BANK_PATH MODEL/nuimages_fsod_10_shots_seed_0/nuim_fsod_query_10_pool7_sel_large.pth \
MODEL.WEIGHT MODEL/mq-glip-l \
TEST.IMS_PER_BATCH 8 VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR results/nuimages_fsod/10_shots_seed_0/model_large_text_and_vision/ 
```

#### FT Free Evaluation (Text only)

```bash
python -m torch.distributed.launch --nproc_per_node 8 tools/test_grounding_net.py \ 
--config-file configs/pretrain/my_configs/mq-glip-l-nuim.yaml \
--additional_model_config configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml \
VISION_QUERY.QUERY_BANK_PATH MODEL/nuimages_fsod_10_shots_seed_0/nuim_fsod_query_10_pool7_sel_large.pth \
MODEL.WEIGHT MODEL/mq-glip-l \
TEST.IMS_PER_BATCH 8 VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR results/nuimages_fsod/10_shots_seed_0/model_large_text_only/ VISION_QUERY.ENABLED False
```

#### FT Free Evaluation (Vision only)

```bash
python -m torch.distributed.launch --nproc_per_node 8 tools/test_grounding_net.py \ 
--config-file configs/pretrain/my_configs/mq-glip-l-nuim.yaml \
--additional_model_config configs/vision_query_10shot/nuimages/nuimages_10_shots_seed0_fsod.yaml \
VISION_QUERY.QUERY_BANK_PATH MODEL/nuimages_fsod_10_shots_seed_0/nuim_fsod_query_10_pool7_sel_large.pth \
MODEL.WEIGHT MODEL/mq-glip-l \
TEST.IMS_PER_BATCH 8 VISION_QUERY.NUM_QUERY_PER_CLASS 10 VISION_QUERY.MAX_QUERY_NUMBER 10 DATASETS.FEW_SHOT 10 OUTPUT_DIR results/nuimages_fsod/10_shots_seed_0/model_large_text_only/ VISION_QUERY.MASK_DURING_INFERENCE True VISION_QUERY.TEXT_DROPOUT 1.0
```





### Acknowledgements
We thank the authors of the [MQDet Repository](https://github.com/YifanXu74/MQ-Det) for their open-source implementations of the MQDet method. This repository is an adaptation of the MQDet codebase to support Foundational FSOD on the nuImages dataset.

### License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


