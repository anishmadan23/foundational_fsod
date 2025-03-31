import os
import numpy as np 
import csv
import glob
import zipfile 
import pickle
from collections import defaultdict
from typing import List, Dict, Any
import json


def group_coco_predictions(flat_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert flat COCO-style predictions into grouped format by image_id,
    and subtract 1 from each predicted category_id to match 0-based class indexing.

    Args:
        flat_preds (List[Dict]): List of predictions in COCO format, where each prediction is a dict:
            {
                "image_id": int,
                "category_id": int,
                "bbox": [x, y, width, height],
                "score": float
            }

    Returns:
        List[Dict]: Grouped predictions in the format:
            [
                {
                    "image_id": int,
                    "instances": [ ... predictions for that image_id ... ]
                },
                ...
            ]
    """
    grouped = defaultdict(list)
    for pred in flat_preds:
        new_pred = pred.copy()
        new_pred["category_id"] = int(pred["category_id"]) - 1  # shift to 0-based for detectron2 based coco evaluation
        grouped[pred["image_id"]].append(new_pred)

    output = [{"image_id": image_id, "instances": instances} for image_id, instances in grouped.items()]
    return output


def main(datasets_links_filepath, results_path):
    dataset_names = []
    with open(datasets_links_filepath, "r") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx>0: # skip header
                url = row[0]
                dataset_names.append(url.split('/')[-2])

    expt_types = ['model_large_text_and_vision', 'model_large_text_only', 'model_large_vision_only']

    dset_pred_map = {}
    for expt in expt_types:
        dset_pred_map[expt] = {}

        for dset in dataset_names:
            print(dset)
            # import ipdb; ipdb.set_trace()
            try:
                path_to_result_file = glob.glob(f"{results_path}/{dset}/{expt}/*/**/bbox.json", recursive=True)[0]   # COCO style preds saved 
            except:
                print(f"Error: No predictions file found for {dset} in {expt}.")
                continue
            
            with open(path_to_result_file, 'r') as f:
                flat_coco_preds = json.load(f)

            grouped_preds = group_coco_predictions(flat_coco_preds)
            save_path = os.path.dirname(path_to_result_file)

            with open(os.path.join(save_path,"bbox.pkl"), "wb") as f:
                pickle.dump(grouped_preds, f)

            # saved_pickle_file_path = convert_to_competition_fmt(path_to_result_file, os.path.dirname(path_to_result_file))
            dset_pred_map[expt][dset] = os.path.join(save_path,"bbox.pkl")
    
     # Define zip file path
    for expt in expt_types:
        zip_path = os.path.join(results_path,f'mqdet_{expt}_{os.path.basename(results_path)}.zip')

        # Create and add files to the zip
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for dset, file_path in dset_pred_map[expt].items():
                # Rename file inside the ZIP (e.g., dataset1_predictions.pkl)
                zip_file_name = f"{dset}.pkl"
                zipf.write(file_path, zip_file_name)  # Add to zip

    # return zip_path  # Return path to the created zip file

    
if __name__=='__main__':
    dataset_type = 'rf20vl_fsod'
    datasets_links_filepath = 'datasets_links.csv'
    results_path = f'/data3/anishmad/msr_thesis/rf_fsod_baselines/mqdet_results/{dataset_type}'

    main(datasets_links_filepath, results_path)
    

        