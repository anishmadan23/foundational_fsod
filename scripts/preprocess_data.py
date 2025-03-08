

import argparse
import json
import os
import random
import numpy as np 
import time
from collections import defaultdict
from copy import deepcopy
import glob
import subprocess
import cv2
import matplotlib.pyplot as plt
import shutil
import requests
from pathlib import Path 
from typing import List, Dict, Tuple
import csv
import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('--rf_url', type=str, default='https://universe.roboflow.com/ds/8Cidf5LjXe?key=zTLWWIsd29')
argparser.add_argument('--dataset_name', type=str, default='liver_disease')
argparser.add_argument('--num_shots', type=int, default=10)
argparser.add_argument('--mode', type=str, default='train')
argparser.add_argument('--visualize', type=bool, default=False)
argparser.add_argument('--min_instance_count', type=int, default=50)
args= argparser.parse_args()

API_KEY = os.environ["ROBOFLOW_API_KEY"]

class PreprocessData:
    def __init__(
            self,
            url,
            modes,
            num_shots=10,
            visualize=False,
        ):
        self.rf_dataset_url = self.get_rf_download_link_from_url(url)
        print(self.rf_dataset_url)
        self.dataset_name = self.get_dataset_name_from_url(url)
        self.visualize = visualize

        self.data_dir = Path(f"data/{self.dataset_name}")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.modes = modes
        self.num_shots = num_shots
        self.save_dir = Path(f'gen_data/{self.dataset_name}')
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # self.min_instance_count = min_instance_count

    def get_dataset(self):
        """
        Download the dataset from Roboflow
        """

        zip_file = os.path.join(self.data_dir, "roboflow.zip")
        if os.path.exists(zip_file):
            return self.data_dir
        try:
            # Step 1: Download the file
            print("Downloading the file...")
            subprocess.run(["curl", "-L", self.rf_dataset_url, "-o", zip_file], check=True)
            
            # Step 2: Unzip the file
            print("Unzipping the file...")
            subprocess.run(["unzip", "-o", zip_file, "-d", self.data_dir], check=True)
            
            
            print("Operation completed successfully.")
            return self.data_dir
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running a subprocess: {e}")
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    

    def get_clean_ann_data(self, data_ann_file):
        
        """
        Get clean annotation data: Removes class 0 that is added by default to Roboflow datasets and shifts annotation ids by 1
        """
        data_ann = json.load(open(data_ann_file, 'r'))
        new_data_ann = {}
        if data_ann['info']:
            new_data_ann['info'] = data_ann['info']
        if data_ann['licenses']:
            new_data_ann['licenses'] = data_ann['licenses']

        # confirm if category 0 is none
        assert(data_ann['categories'][0]['supercategory']=='none'), "Need to change logic for removing category 0 from dataset in preprocessing"

        # data_ann['categories'] = [cat for cat in data_ann['categories'] if cat['id']!=0]
        new_data_ann['categories'] = [{'id': cat['id']-1, 'name': cat['name'], 'supercategory': cat['supercategory']} for cat in data_ann['categories'] if cat['id']!=0]

        new_data_ann['images'] = data_ann['images']
        new_data_ann['annotations'] = deepcopy(data_ann['annotations'])

        for ann in new_data_ann['annotations']:
            ann['category_id'] = ann['category_id']-1

        category_id_counts = defaultdict(int)
        for ann in new_data_ann['annotations']:
            category_id_counts[ann['category_id']] += 1

        # category_ids_to_remove = {cat_id for cat_id, count in category_id_counts.items() if count < self.min_instance_count}

        # filtered_annotations = [ann for ann in new_data_ann['annotations'] if ann['category_id'] not in category_ids_to_remove]
        # new_data_ann['annotations'] = filtered_annotations

        # new_data_ann['categories'] = [cat for cat in new_data_ann['categories'] if cat['id'] not in category_ids_to_remove]

        img_ids_with_annotations = {ann['image_id'] for ann in new_data_ann['annotations']}
        new_data_ann["images"] = [img for img in new_data_ann["images"] if img['id'] in img_ids_with_annotations]

        ID2CLASS = {}
        for cat_info in new_data_ann['categories']:
            ID2CLASS[cat_info['id']] = cat_info['name']

        return new_data_ann, ID2CLASS
    

    def get_save_path_seeds(self, cls, shots, seed):
        """
        Get save path for the tmp split
        """

        prefix = "full_box_{}_shots_{}".format(shots, cls)
        save_dir = self.save_dir / "tmp_splits" / self.mode / f"seed{seed}"
        save_dir.mkdir(exist_ok=True, parents=True)

        save_path = save_dir / f"{prefix}.json"
        return save_path


    # def modify_dataset_for_few_shot(self, dataset, base_split_path, shots=5, seed=1, valmode=False):
        
    #     if valmode:
    #         all_split_files = glob.glob(os.path.join(base_split_path, 'val', 'seed'+str(seed), f'full_box_{shots}shot_*.json'))
    #     else:
    #         all_split_files = glob.glob(os.path.join(base_split_path, 'seed'+str(seed), f'full_box_{shots}shot_*.json'))
            
    #     new_dataset_dict = {}
    #     for cls_split_file in all_split_files:
    #         # get matching info from dataset and add it to a new list which would work as new FS dataset
    #         cls_info = json.load(open(cls_split_file, 'r'))
    #         for idx_ann, cls_info_ann in enumerate(cls_info['annotations']):
    #             img_id = cls_info_ann['image_id']
    #             all_data_info = dataset[img_id]                              # corresponding info in all data 
    #             assert(img_id==all_data_info['image_id']), f"Image id of split file is {img_id} and its corresponding one in all dataset is {all_data_info['image_id']}"
    #             match_flag=0
    #             for all_data_info_ann in all_data_info['annotations']:
    #                 if cls_info_ann['bbox'] == all_data_info_ann['bbox']:
    #                     match_flag=1
    #                     if img_id in new_dataset_dict:
    #                         new_dataset_dict[img_id]['annotations'].append(all_data_info_ann)
    #                     else:
    #                         new_dataset_dict[img_id] = deepcopy(all_data_info)
    #                         new_dataset_dict[img_id]['annotations'] = [all_data_info_ann]          # replace all image annotations by the split annotation only.                    break
    #             # if match_flag==0:
    #             assert(match_flag==1), "check annotation as bbox match was not found"

    #     new_dataset = list(new_dataset_dict.values())
        
    #     return new_dataset
    
    def visualize_best_split(self, best_split_file):
        output_dir = self.save_dir / "visualization" / self.mode
        output_dir.mkdir(exist_ok=True, parents=True)

        data = json.load(open(best_split_file, 'r'))
        image_id_to_file = {img['id']: img['file_name'] for img in data['images']}

        # Map category IDs to category names
        category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

        annotations_by_image = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)

        # Iterate over images and their annotations
        for image_id, annotations in annotations_by_image.items():
            image_file = str(self.data_dir / self.mode / image_id_to_file.get(image_id))

            if image_file and os.path.exists(image_file):
                # Load image
                img = cv2.imread(image_file)
                if img is None:
                    continue

                # Convert BGR to RGB for visualization
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Organize annotations by class
                annotations_by_class = {}
                for annotation in annotations:
                    category_id = annotation['category_id']
                    category_name = category_id_to_name.get(category_id, 'Unknown')
                    if category_name not in annotations_by_class:
                        annotations_by_class[category_name] = []
                    annotations_by_class[category_name].append(annotation)

                # Save visualized images for each class
                for class_name, class_annotations in annotations_by_class.items():
                    img_copy = img.copy()  # Work on a copy for each class

                    # Draw bounding boxes and labels for the current class
                    for annotation in class_annotations:
                        bbox = annotation['bbox']
                        x, y, width, height = bbox
                        x, y, width, height = int(x), int(y), int(width), int(height)

                        # Draw bounding box
                        cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

                        # Prepare text background
                        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        text_width, text_height = text_size[0], text_size[1]
                        text_x, text_y = x, y - 10
                        cv2.rectangle(img_copy, (text_x, text_y - text_height - 4), (text_x + text_width, text_y + 2), (255, 255, 255), -1)

                        # Put label
                        cv2.putText(img_copy, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Save the image in the class-specific folder
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    output_path = os.path.join(class_dir, os.path.basename(image_file))
                    plt.imsave(output_path, img_copy)
    
    # def get_rf_download_link_from_url(self, dataset_url):
    #     # dataset_url = "https://universe.roboflow.com/roboflow-100/activity-diagrams-qdobr/dataset/1"
    #     dataset_url = dataset_url.replace("/dataset/", "/")
    #     url = os.path.join(dataset_url, "coco")
    #     url = url.replace("https://universe", "https://api")
    #     url = url.replace("https://app", "https://api")
    #     print("Getting download link from", url)
    #     response = requests.get(url, params={"api_key": API_KEY})
    #     response.raise_for_status()
    #     print(response.json())
    #     link = response.json()["export"]["link"]
    #     print('link', link)
    #     return link


    def get_rf_download_link_from_url(self, dataset_url):
        # Modify the URL to target the API endpoint
        dataset_url = dataset_url.replace("/dataset/", "/")
        url = os.path.join(dataset_url, "coco")
        url = url.replace("https://universe", "https://api")
        url = url.replace("https://app", "https://api")
        print("Getting download link from", url)

        max_retries = 5
        delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            response = requests.get(url, params={"api_key": API_KEY})
            response.raise_for_status()
            data = response.json()
            print("API response:", data)

            if "export" in data:
                link = data["export"]["link"]
                print("Download link:", link)
                return link
            else:
                print(f"Attempt {attempt}/{max_retries}: Export link not ready yet. Retrying in {delay} seconds...")
                time.sleep(delay)

        raise KeyError("Export link not found in API response after multiple retries.")


    def get_dataset_name_from_url(self, dataset_url):
        return dataset_url.split("/")[-2]


def is_coco_format(file_path):
    """
    Check if a JSON file is in COCO format.
    
    Args:
        file_path (str): Path to the JSON file to be checked.
        
    Returns:
        bool: True if the file is in COCO format, False otherwise.
        str: Explanation of the validation result.
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check for required keys
        required_keys = {"images", "annotations", "categories"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Validate "images" is a list of dictionaries with necessary fields
        if not isinstance(data["images"], list):
            return False, "'images' should be a list."
        if not all(isinstance(img, dict) and "id" in img and "file_name" in img for img in data["images"]):
            return False, "'images' should contain dictionaries with at least 'id' and 'file_name' fields."
        
        # Validate "annotations" is a list of dictionaries
        if not isinstance(data["annotations"], list):
            return False, "'annotations' should be a list."
        if not all(isinstance(ann, dict) and "id" in ann and "image_id" in ann and "category_id" in ann for ann in data["annotations"]):
            return False, "'annotations' should contain dictionaries with at least 'id', 'image_id', and 'category_id' fields."
        
        # Validate "categories" is a list of dictionaries
        if not isinstance(data["categories"], list):
            return False, "'categories' should be a list."
        if not all(isinstance(cat, dict) and "id" in cat and "name" in cat for cat in data["categories"]):
            return False, "'categories' should contain dictionaries with at least 'id' and 'name' fields."
        
        # All checks passed
        return True, "The file is in valid COCO format."
    
    except json.JSONDecodeError:
        return False, "Invalid JSON format."
    except FileNotFoundError:
        return False, "File not found."
    except Exception as e:
        return False, f"Unexpected error: {e}"

class FewShotException(Exception):
    pass


def main(url, modes, num_shots=10, visualize=False):
    
    prepro_data= PreprocessData(url, num_shots=num_shots, modes=modes)
    ## 1. download and extract dataset

    dataset_dir  = prepro_data.get_dataset()
    # path to save best split

    for mode in modes:
        save_new_anno_path = prepro_data.save_dir / "fsod_data_detectron" / mode 
        save_new_anno_path.mkdir(exist_ok=True, parents=True)
        best_split_save_path = save_new_anno_path / f"{prepro_data.dataset_name}_fsod_{mode}_best_split_shots_{prepro_data.num_shots}.json"


        clean_annos_test,_ = prepro_data.get_clean_ann_data(dataset_dir / mode / "_annotations.coco.json")
        with open(best_split_save_path, 'w') as f:
            json.dump(clean_annos_test, f)
    
            # sanity check to confirm if the saved file is in COCO format
        is_valid, message = is_coco_format(best_split_save_path)
        if not is_valid:
            raise FewShotException(f"Saved file is not in COCO format. Message: {message}")
        print(f"Is COCO format: {is_valid}, Message: {message}")

            ## 5. Visualize the best split
        if prepro_data.visualize:
            prepro_data.visualize_best_split(best_split_save_path)
            


if __name__ == '__main__':
    links_csv = "datasets_links.csv"
    links = []
    with open(links_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            url = row[0]
            links.append(url)
    modes = ['train', 'valid','test']
    links = links[1:]
    for url in tqdm.tqdm(links):
        # for mode in :
        # manually copy the train split to val split as that is empty -- hacky workaround
        # should have no effect for the zero-shot baseline
        # if url == 'https://app.roboflow.com/rf20vl/gwhd2021-fsod-oxon/1':  
        #     main(
        #     url,
        #     num_shots = 10,
        #     modes=['train', 'test'],
        #     visualize=False,
        #     # min_instance_count=args.min_instance_count,
        # )
        # else:
        main(
            url,
            num_shots = 10,
            modes=modes,
            visualize=False,
            # min_instance_count=args.min_instance_count,
        )
