import os
import json
import cv2
import matplotlib.pyplot as plt

# Load JSON file
json_path = 'liver_disease_fsod_train_seed_0_shots_10.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Map image IDs to file paths
image_id_to_file = {img['id']: img['file_name'] for img in data['images']}

# Map category IDs to category names
category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# Create output directory for visualizations
output_dir = 'visualized_images'
os.makedirs(output_dir, exist_ok=True)

# Group annotations by image ID
annotations_by_image = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(annotation)

# Iterate over images and their annotations
for image_id, annotations in annotations_by_image.items():
    image_file = image_id_to_file.get(image_id).replace("/data3/anishmad/roboflow_data/", "")

    if image_file and os.path.exists(image_file):
        # Load image
        img = cv2.imread(image_file)
        if img is None:
            continue

        # Convert BGR to RGB for visualization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw all bounding boxes for the image
        for annotation in annotations:
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            category_name = category_id_to_name.get(category_id, 'Unknown')

            # Draw bounding box
            x, y, width, height = bbox
            x, y, width, height = int(x), int(y), int(width), int(height)
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Prepare text background
            text_size = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_width, text_height = text_size[0], text_size[1]
            text_x, text_y = x, y - 10
            cv2.rectangle(img, (text_x, text_y - text_height - 4), (text_x + text_width, text_y + 2), (255, 255, 255), -1)

            # Put label
            cv2.putText(img, category_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save visualized image
        output_path = os.path.join(output_dir, os.path.basename(image_file))
        plt.imsave(output_path, img)

print(f"Visualized images saved to '{output_dir}'")
