# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def explore_dataset(data_dir, output_dir):
#     for split in ['train', 'val']:
#         split_dir = os.path.join(data_dir, split)
        
#         # Explore front, rear, left, right, bev, and homography images
#         for img_type in ['front', 'rear', 'left', 'right', 'bev', 'homography']:
#             img_dir = os.path.join(split_dir, img_type, img_type)  # Note the double 'img_type'
#             image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
#             if image_files:
#                 sample_image = plt.imread(os.path.join(img_dir, image_files[0]))
                
#                 plt.figure(figsize=(10, 10))
#                 plt.imshow(sample_image)
#                 plt.title(f"{split} - {img_type} sample")
#                 plt.savefig(os.path.join(output_dir, f"{split}_{img_type}_sample.png"))
#                 plt.close()
#             else:
#                 print(f"No PNG files found in {img_dir}")

# if __name__ == "__main__":
#     data_dir = "/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR"
#     output_dir = "/home/paperspace/Projects/360CameraToBirdsEyeView/visuals"
#     os.makedirs(output_dir, exist_ok=True)
#     explore_dataset(data_dir, output_dir)

import torch
from torch.utils.data import DataLoader
from preprocessing import Cam2BEVDataset
import numpy as np

def analyze_dataset(dataset_path):
    dataset = Cam2BEVDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    for _, labels in dataloader:
        all_labels.append(labels.numpy())
    
    all_labels = np.concatenate(all_labels)
    unique_labels = np.unique(all_labels)
    
    print(f"Unique label values: {unique_labels}")
    print(f"Min label value: {np.min(unique_labels)}")
    print(f"Max label value: {np.max(unique_labels)}")
    print(f"Number of unique labels: {len(unique_labels)}")

if __name__ == "__main__":
    train_path = "/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/train"
    analyze_dataset(train_path)