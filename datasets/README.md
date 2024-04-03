# Prepare datasets
We follow the basic guidelines provided in the [Detic codebase](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md) to setup a new few-shot dataset. We describe how to setup [nuImages](https://nuscenes.org/nuimages) and [LVIS v0.5](https://www.lvisdataset.org/) below. To use a different custom dataset, follow the [detectron2 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) to register a new dataset, and modify the few-shot split generation script accordingly.

## nuImages
1.  **Download NuImages** : First, download the nuImages dataset and place/soft-link it in `nuimages`. We provide COCO-style [annotation files here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/nuimages_coco_fmt/annotations) for easy use with the Detectron2 codebases. To create these annotation files from scratch, please refer to the nuImages mmdetection3d data creation script (specifically by running [this file](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/nuimage_converter.py); follow the instructions [here](https://mmdetection3d.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-conversion))
```
$REPOSITORY_ROOT/data/datasets/
    nuimages/
        images/
            samples/
        annotations/
            nuimages_v1.0-train.json
            nuimages_v1.0-val.json
```

2. **Pre-Process Dataset**: Next, pre-process the annotation file to remove classes with extremely few samples (e.g. we remove *wheelchair* (id=13))

```
python data/scripts/preprocess_nuimages.py 
```

- The generated annotation file would be saved in `$REPOSITORY_ROOT/datasets/nuimages/annotations/no_wc/`

3. **Generate Few-Shot Split Files** : Next, generate few-shot splits from the resulting training annotation file, if not already available. We provide the [few-shot splits here](https://huggingface.co/anishmadan23/foundational_fsod/tree/main/data_splits/nuimages/10_seeds), so no need to generate these. For a custom dataset, one would have to generate these files and convert it to the COCO format.


4. **Register Few Shot Datasets**: For a  ***custom datasets*** or ***new few-shot split files*** (e.g. NOT nuImage), change paths for the few-shot files in `detectron2/data/datasets/builtin.py` accordingly. See `datasets/CUSTOM_DATASET_SUPPORT_README.md` for further details on how to register a new dataset.



### Metadata
We use LVIS as our running example, but these instructions can be easily adapted for nuImages.
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

`lvis_v1_train_cat_info.json` is used by FedLoss.
This is created by running 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each datasets.
They are created by running
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

