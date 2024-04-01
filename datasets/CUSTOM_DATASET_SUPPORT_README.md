# Adding support for new dataset in Detic

### Step 1
In `detectron2/data/datasets/builtin.py`, add path information relevant to new dataset. Note that the dataset should be in COCO format for the following steps. For example, consider adding nuscenes dataset support: 

```python
_PREDEFINED_SPLITS_NUSCENES = {}
_PREDEFINED_SPLITS_NUSCENES["nuscenes_all_cls"] = {
    
    "nuscenes_all_cls_train": ("<path_to_images>", "<train_annotation_file_path>"),
    "nuscenes_all_cls_val": ("<path_to_images>", "<val_annotation_file_path>"),
}

```
Make a new function to register the custom dataset:
```python
def register_all_nuscenes(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUSCENES.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )
```

Towards the end of script add the following line:
```python
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
   ...                                 # other calls to register different datasets
    register_all_nuscenes(_root)    # add this line

```

### Step 2

Edit the `_get_builtin_metadata(dataset_name)` in `detectron2/data/datasets/builtin_meta.py` to add information about categories in the new dataset and some other information

```python
...
    elif dataset_name == 'nuscenes_all_cls':
        categories = [
            {"id": 0, "name": "car"}, 
            {"id": 1, "name": "truck"}, 
            {"id": 2, "name": "trailer"}, 
            {"id": 3, "name": "bus"}, 
            {"id": 4, "name": "construction_vehicle"},
            {"id": 5, "name": "bicycle"}, 
            {"id": 6, "name": "motorcycle"},
            {"id": 7, "name": "emergency_vehicle"},
            {"id": 8, "name": "adult"}, 
            {"id": 9, "name": "child"},
            {"id": 10, "name": "police_officer"},
            {"id": 11, "name": "construction_worker"},
            {"id": 12, "name": "stroller"},
            {"id": 13, "name": "personal_mobility"},
            {"id": 14, "name": "pushable_pullable"},
            {"id": 15, "name": "debris"},
            {"id": 16, "name": "traffic_cone"},
            {"id": 17, "name": "barrier"}
        ]

        id_to_name = {x['id']: x['name'] for x in categories}
        thing_dataset_id_to_contiguous_id = {i: i for i in range(len(categories))}
        thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes}
```

### Step 3: Dump CLIP features according to new dataset categories

Before running Detic on the new dataset, we need to make a new config. This requires generating the clip features for new dataset categories. Use the following command to dump clip features:

```python
python tools/dump_clip_features.py --ann <path_to_COCO_fmt_annotation_file> --out_path ./datasets/metadata/nuscenes_all_cls.npy

```
Change config arguments:
```yaml
MODEL:
  TEST_CLASSIFIERS: ("datasets/metadata/nuscenes_all_cls.npy",)
  TEST_NUM_CLASSES: [18,]
  ```
Also change other information and paths according to new dataset in the config. The training might not end gracefully if segmentation information is not present but the detection results will be saved.