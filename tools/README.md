## Key commands to run

### Generating file which computes image count for a particular class (Useful for federated loss expts)
```python tools/get_lvis_cat_info.py --ann <Path to nuimages train or lvis train annotation file>```


- To generate image count for only rare categories of LVIS, use appropriate annotation file:
```python tools/get_lvis_cat_info.py --ann data/datasets/lvis/my_data/lvis_v1_trainval_novel.json```

### To generate CLIP features (used as clf weights)
```python tools/dump_clip_features.py --ann data/datasets/lvis/lvis_v1_train.json  --out_path datasets/metadata/lvis_v1_all_cats.npy```
