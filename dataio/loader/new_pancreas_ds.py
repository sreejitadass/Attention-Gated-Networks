import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
import pydicom

class PancreasDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, preload_data=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.preload_data = preload_data

        # Find DICOM files
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*/*/*/*.dcm"), recursive=True))

        if not self.image_paths:
            raise FileNotFoundError(f"No DICOM files found in {root_dir}. Check dataset structure.")

        # Preload labels if enabled
        if self.preload_data:
            self.preloaded_labels = [self.load_label(i) for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.load_image(index)
        label = self.load_label(index)

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Convert to PyTorch tensor
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def load_image(self, index):
        """ Loads and preprocesses DICOM image. """
        dicom_path = self.image_paths[index]
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        # Normalize Image
        image = (image - image.min()) / (image.max() - image.min())

        return image

    def load_label(self, index):
        """
        Generates or retrieves the label for an image.
        Ensures label matches image dimensions: [1, 512, 512].
        """
        dicom_path = self.image_paths[index]
        label = self.get_label_from_path(dicom_path)

        # Ensure label is a NumPy array
        if isinstance(label, (int, float)):
            label = np.full((512, 512), label, dtype=np.uint8)  # Match image size
        elif isinstance(label, np.ndarray):
            if label.shape == ():  
                label = np.full((512, 512), label.item(), dtype=np.uint8)
            elif label.shape != (512, 512):  
                raise ValueError(f"Unexpected label shape {label.shape}, expected (512, 512).")

        # Convert to PyTorch tensor, add batch dimension
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)  # Shape: [1, 512, 512]

        return label

    def get_label_from_path(self, path):
        """ Placeholder function to assign labels based on file structure. """
        return 1  # Modify if you have actual label logic
