import roboflow
from roboflow.core.project import Project
from tqdm import tqdm
from typing import List
import os
import shutil
import json
from copy import deepcopy

fsod_api_key = os.getenv("ROBOFLOW_API_KEY")
rf_fosd = roboflow.Roboflow(api_key=fsod_api_key)

# change workspace name to the datasets you want to download:
# rf20-vl-fsod is the workspace with the 20 fsod datasets for the CVPR 2025 challenge currently running
# rf100-vl-fsod is the workspace with the 100 fsod datasets, of which the 20 are a subset
# rf100-vl is the workspace with the same 100 datasets as above, but they aren't few-shot in nature.

fsod_workspace = rf_fosd.workspace("rf20-vl-fsod")
projects_array: List[Project] = []
fails = 0
print(fsod_workspace.project_list)
for a_project in fsod_workspace.project_list:
    proj = Project(fsod_api_key, a_project, fsod_workspace.model_format)
    projects_array.append(proj)

data_dir = os.path.join("scratch", "tmp_data")
os.makedirs(data_dir, exist_ok=True)

urls = []
from concurrent.futures import ThreadPoolExecutor

def generate_project(project):
    if project.versions():
        return 0
    v1 = project.generate_version({"preprocessing": {"auto-orient": True}, "augmentation": {}})
    return 1

def process_project(project):
    last_version = max(project.versions(), key=lambda v: v.id)
    download_dir = os.path.join(data_dir, project.name)
    # downloaded = last_version.download(model_format="coco", location=download_dir)
    coco_url = last_version._Version__get_download_url("coco")
    universe_url = coco_url.replace("coco", "")
    universe_url = universe_url.replace("https://api", "https://universe")
    universe_url = universe_url.split("/")
    # universe_url.insert(-2, "dataset")
    universe_url = "/".join(universe_url)
    #drop last /
    universe_url = universe_url[:-1]
    return universe_url

with ThreadPoolExecutor(max_workers=12) as executor:
    for url in tqdm(executor.map(process_project, projects_array), total=len(projects_array)):
        urls.append(url)
    
with open("datasets_links.csv", "w") as f:
    f.write("Link (Link to version)\n")
    for url in urls:
        f.write(url + "\n")