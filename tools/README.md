## Key commands to run

### Generating file which computes image count for a particular class (Useful for federated loss expts)
```python tools/get_lvis_cat_info.py --ann <Path to nuimages train or lvis train annotation file>```


- To generate image count for only rare categories of LVIS, use appropriate annotation file:
```python tools/get_lvis_cat_info.py --ann data/datasets/lvis/my_data/lvis_v1_trainval_novel.json```

### To generate CLIP features (used as clf weights)
```python tools/dump_clip_features.py --ann data/datasets/lvis/lvis_v1_train.json  --out_path datasets/metadata/lvis_v1_all_cats.npy```

### To generate few-shot splits
```python tools/generate_few_shots_splits.py --data_train_path <train_ann_file> --base_save_path <> --dset_name <>```

- To convert the individual split files to detectron2 format (as used by Foundational FSOD):
  - Add [image dir and annotation file path](https://github.com/anishmadan23/detectron2-ffsod/blob/main/detectron2/data/datasets/builtin.py#L50-L56) in detectron2.
  - uncomment the lines towards the end in the above script.
  