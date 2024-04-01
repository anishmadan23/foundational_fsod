# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import clip

def compute_clip_features(ann_file, custom_vocabulary=None, clip_model="ViT-B/32", prompt='a', fix_space=False, use_underscore=False):
    if custom_vocabulary is None:
        data = json.load(open(ann_file, 'r'))
        cat_names = [x['name'] for x in \
            sorted(data['categories'], key=lambda x: x['id'])]
        
    else:
        cat_names = [x['name'] for x in \
            sorted(custom_vocabulary, key=lambda x: x['id'])]
    if fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synonyms = []
    if prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
            for x in synonyms]
    elif prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
            for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
        sum(len(x) for x in sentences_synonyms))

    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device=device)
    # if args.avg_synonyms:
    #     sentences = list(itertools.chain.from_iterable(sentences_synonyms))
    #     print('flattened_sentences', len(sentences))
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        if len(text) > 10000:
            text_features = torch.cat([
                model.encode_text(text[:len(text) // 2]),
                model.encode_text(text[len(text) // 2:])],
                dim=0)
        else:
            text_features = model.encode_text(text)
    print('text_features.shape', text_features.shape)
    # if args.avg_synonyms:
    #     synonyms_per_cat = [len(x) for x in sentences_synonyms]
    #     text_features = text_features.split(synonyms_per_cat, dim=0)
    #     text_features = [x.mean(dim=0) for x in text_features]
    #     text_features = torch.stack(text_features, dim=0)
    #     print('after stack', text_features.shape)
    text_features = text_features.cpu().numpy()
    return text_features, cat_names
    # if args.out_path != '':
    #     print('saveing to', args.out_path)
    #     np.save(open(args.out_path, 'wb'), text_features)
    # import pdb; pdb.set_trace()

