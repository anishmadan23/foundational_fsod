import os
import numpy as np 
import csv
import glob
import zipfile

def main(datasets_links_filepath, results_path):
    dataset_names = []
    with open(datasets_links_filepath, "r") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx>0: # skip header
                url = row[0]
                dataset_names.append(url.split('/')[-2])

    dset_pred_map = {}
    for dset in dataset_names:
        print(dset)
        path_to_result_file = glob.glob(f"{results_path}/{dset}/*/**/instances_predictions.pkl", recursive=True)[0]
        dset_pred_map[dset] = path_to_result_file
    
     # Define zip file path
    zip_path = os.path.join(results_path,'predictions_rf20vl_zs.zip')

    # Create and add files to the zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dset, file_path in dset_pred_map.items():
            # Rename file inside the ZIP (e.g., dataset1_predictions.pkl)
            zip_file_name = f"{dset}.pkl"
            zipf.write(file_path, zip_file_name)  # Add to zip

    return zip_path  # Return path to the created zip file

    
if __name__=='__main__':
    datasets_links_filepath = 'datasets_links.csv'
    results_path = '/data3/anishmad/msr_thesis/rf_fsod_baselines/evalai_baselines/'

    main(datasets_links_filepath, results_path)
    

        