import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class BEVDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(256, 256)):
        """
        Initialize the BEVDataset.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Data split ('train', 'val', 'test').
            img_size (tuple): Size to which images will be resized.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),  # Resize the image
            transforms.ToTensor(),  # Convert image to PyTorch tensor
        ])

        # Load all samples
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} set")

    def _load_samples(self):
        """
        Load all valid samples from the dataset.

        Returns:
            list: List of dictionaries containing paths to input views and BEV target.
        """
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        view_dirs = ['front', 'rear', 'left', 'right']

        # Iterate through all files in the front view directory
        for filename in os.listdir(os.path.join(split_dir, 'front', 'front')):
            if filename.endswith('.png'):
                # Create a sample dictionary with paths to all views and BEV target
                sample = {
                    'views': [os.path.join(split_dir, view, view, filename) for view in view_dirs],
                    'bev': os.path.join(split_dir, 'bev', 'bev', filename)
                }
                
                # Check if all required files exist
                if all(os.path.exists(path) for path in sample['views'] + [sample['bev']]):
                    samples.append(sample)
                else:
                    print(f"Skipping {filename} due to missing files")

        return samples

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing input views and BEV target tensors.
        """
        sample = self.samples[idx]

        # Load and process input views
        views = [self.transform(Image.open(view_path)) for view_path in sample['views']]
        inputs = torch.cat(views, dim=0)  # Concatenate along channel dimension

        # Load and process BEV target
        bev_target = self.transform(Image.open(sample['bev']))

        return {'inputs': inputs, 'bev_target': bev_target}

def get_dataloader(root_dir, split='train', batch_size=4, num_workers=2, img_size=(256, 256)):
    """
    Create a DataLoader for the BEVDataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Data split ('train', 'val', 'test').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        img_size (tuple): Size to which images will be resized.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Create a BEVDataset instance
    dataset = BEVDataset(root_dir, split, img_size)

    # Create and return a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),  # Shuffle only if it's the training set
        num_workers=num_workers
    )

    return dataloader