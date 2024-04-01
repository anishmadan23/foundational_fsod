# Prepare datasets
We follow the basic guidelines provided in the [Detic codebase](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md) to setup a dataset. We describe how to setup 2 datasets: [nuImages](https://nuscenes.org/nuimages) and [LVIS v0.5](https://www.lvisdataset.org/) . Using a custom dataset should be relatively simple: follow [detectron2 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) to register a new dataset, and modify few-shot split generation script accordingly.

## nuImages
1.  **Download NuImages** : First, download nuImages dataset and place/soft-link it in `nuimages` as follows. We provide the [annotation files here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/nuimages_coco_fmt/annotations) converted in the COCO format for easy use with Detectron2 codebases. If the files are not accessible for some reason, one could also use convert to this format by using mmdetection3d(specifically by running [this file](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/nuimage_converter.py); follow the instructions [here](https://mmdetection3d.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-conversion))
```
$REPOSITORY_ROOT/data/datasets/
    nuimages/
        images/
            samples/
        annotations/
            nuimages_v1.0-train.json
            nuimages_v1.0-val.json
```

2. **Pre-Process Dataset**: Next, pre-process the annotation file to remove classes with extremely few samples(only wheelchair class (id=13) to be removed)

```
python data/scripts/preprocess_nuimages.py 
```

- The generated annotation file would be saved in `$REPOSITORY_ROOT/datasets/nuimages/annotations/no_wc/`

3. **Generate Few-Shot Split Files** : Next, generate few-shot splits from the resulting training annotation file, if not already available. We provide the [few-shot splits here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/data_splits/nuimages/10_seeds), so no need to generate these. For a custom dataset, one would have to generate these files and convert it to the COCO format.


4. **Register Few Shot Datasets**: Only do this step for ***custom datasets*** or ***new few-shot split files***, as we provide support for registering nuimages few-shot datasets. Be sure to change paths for the few-shot files in `detectron2/data/datasets/builtin.py` corresponding to your own. See `datasets/CUSTOM_DATASET_SUPPORT_README.md` for an idea of how to register a new dataset.



### Metadata

```
metadata/
    lvis_v1_train_cat_info.json
    coco_clip_a+cname.npy
    lvis_v1_clip_a+cname.npy
    o365_clip_a+cnamefix.npy
    oid_clip_a+cname.npy
    imagenet_lvis_wnid.txt
    Objects365_names_fix.csv
```

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each datasets.
They are created by (taking LVIS as an example)
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
~~~
Note we do not include the 21K class embeddings due to the large file size.
To create it, run
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val_lvis-21k.json --out_path datasets/metadata/lvis-21k_clip_a+cname.npy
~~~


### TODOS
- [x] Release data splits 
- [ ] Data split generation code
- [ ] Data split conversion code (for detectron compatibility)

