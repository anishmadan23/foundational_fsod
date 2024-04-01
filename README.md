# Foundational Few Shot Object Detection (F-FSOD)
[![arXiv](https://img.shields.io/badge/arXiv-2312.14494-b31b1b.svg)](https://arxiv.org/abs/2312.14494)

![teaser.png](assets/teaser.png)

## Abstract
Few-shot object detection (FSOD) benchmarks have advanced techniques for detecting new categories with limited annotations. Existing benchmarks repurpose wellestablished datasets like COCO by partitioning categories into base and novel classes for pre-training and finetuning respectively. However, these benchmarks do not reflect how FSOD is deployed in practice. Rather than only pre-training on a small number of base categories, we argue that it is more practical to fine-tune a foundation model (e.g., a vision-language model (VLM) pre-trained on webscale data) for a target domain. Surprisingly, we find that zero-shot inference from VLMs like GroundingDINO significantly outperforms the state-of-the-art (48.3 vs. 33.1 AP) on COCO. However, such zero-shot models can still be **misaligned** to target concepts of interest. For example,
trailers on the web may be different from trailers in the context of autonomous vehicles. In this work, we propose **Foundational FSOD**, a *new benchmark protocol* that evaluates detectors pre-trained on any external datasets and fine-tuned on K-shots per target class. Further, we note that current FSOD benchmarks are actually federated datasets containing exhaustive annotations for each category on a subset of the data. We leverage this insight to propose simple strategies for fine-tuning VLMs with federated losses. We demonstrate the effectiveness of our approach on LVIS and nuImages, improving over prior work by 5.9 AP.

## Installation
See [installation instructions](docs/INSTALL.md).


## Data
See [datasets/README.md](datasets/README.md)

## Models
Create `models/` in root directory and download pre-trained model used for FSOD training from [here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/pretrained_models/)

## Training
```python
python train_net.py --num-gpus 1 --config-file <config_path>  --pred_all_class  OUTPUT_DIR_PREFIX <root_output_dir>
```

### Config Details
 - Naive Finetuning: [`configs/nuimages_cr/code_release_v2/naive_ft_shots10_seed_0.yaml`](configs/nuimages_cr/code_release_v2/naive_ft_shots10_seed_0.yaml)
 - FedLoss: [`configs/nuimages_cr/code_release_v2/fedloss_num_sample_cats_4_shots10_seed_0.yaml`](configs/nuimages_cr/code_release_v2/fedloss_num_sample_cats_4_shots10_seed_0.yaml)
 - Inverse FedLoss: [`configs/nuimages_cr/code_release_v2/invfedloss_num_sample_cats_4_shots10_seed_0.yaml`](configs/nuimages_cr/code_release_v2/invfedloss_num_sample_cats_4_shots10_seed_0.yaml)
 - PseudoNegatives: [`configs/nuimages_cr/code_release_v2/pseudo_negatives_shots10_seed_0.yaml`](configs/nuimages_cr/code_release_v2/pseudo_negatives_shots10_seed_0.yaml)
 - Deterministic FedLoss or True Negatives (Oracle): [`configs/nuimages_cr/code_release_v2/detfedloss_shots10_seed_0.yaml`](configs/nuimages_cr/code_release_v2/detfedloss_shots10_seed_0.yaml)

 
 ### Key Config Fields
 - `DATASETS.TRAIN`: Specify training split according to registered datasets.
 - `DATASETS.TEST`: Test set to evaluate method on
 - `ROI_BOX_HEAD.FED_LOSS_NUM_CAT`: Num of categories to be sampled for FedLoss
 - `ROI_BOX_HEAD.USE_FED_LOSS`: Flag to enable federated loss
 - `ROI_BOX_HEAD.INVERSE_WEIGHTS`: Flag to enable inverse frequency weights with federated loss
 - `ROI_BOX_HEAD.ALL_ANN_FILE`: Used in sampling strategy for fedloss. If using with pseudonegatives, fill this field with predictions on the few-shot trainset from a teacher model.

 ### PseudoNegatives Training
 1. Train a teacher model T or use the pretrained model available. 
 2. Make a new config to run inference on FSOD trainset, for eg. by setting DATASETS.TEST as `nuimages_fsod_train_seed_0_shots_10`
 3. Convert the generated predictions `.pth` file to COCO format by using `tools/convert_preds_to_ann.py`. Also specify the confidence threshold to filter pseudolabels. See sample command below

```python
python tools/convert_preds_to_ann.py --pred_path_train <path_trainset_eval_pth_file> --dataset_name nuimages_fsod_train_seed_0_shots_10 --conf_thresh 0.2
```

4. Plug the generated file in pseudonegatives config file (in `ROI_BOX_HEAD.ALL_ANN_FILE`)

## Inference

```python 
python train_net.py --num-gpus 8 --config-file <config_path>  --pred_all_class --eval-only  MODEL.WEIGHTS <model_path> OUTPUT_DIR_PREFIX <root_output_dir>
```

## TODO
- [x] Code cleanup 
- [x] Data related support in code
- [x] Release FSOD training files 
- [ ] FSOD Data split creation : nuImages along with new split
- [ ] Release trained model 
- ------------
- [ ] LVIS support in data and training models


## Acknowledgment


## Citation


