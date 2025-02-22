import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
import pydicom
import torch.nn.functional as F  # For label resizing

class PancreasDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, preload_data=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.preload_data = preload_data

        # Find patient folders (exact match for PANCREAS_XXXX)
        print(f"Searching for patient folders in: {root_dir}")
        self.patient_dirs = sorted(glob.glob(os.path.join(root_dir, "PANCREAS_[0-9][0-9][0-9][0-9]")))
        
        # Debug: Print found patient directories
        print(f"Found patient directories: {self.patient_dirs}")

        self.image_paths = []
        for pdir in self.patient_dirs:
            # Search for first-level subfolders (e.g., UID-like folders)
            first_level_subdirs = sorted(glob.glob(os.path.join(pdir, "*")))
            dcm_files = []
            for subdir1 in first_level_subdirs:
                # Search for second-level subfolders
                second_level_subdirs = sorted(glob.glob(os.path.join(subdir1, "*")))
                for subdir2 in second_level_subdirs:
                    # Try multiple patterns for .dcm files (case-insensitive and common variants)
                    dcm_patterns = [
                        os.path.join(subdir2, "*.dcm"),
                        os.path.join(subdir2, "*.DCM"),
                        os.path.join(subdir2, "*.dicom"),
                        os.path.join(subdir2, "*.DICOM"),
                    ]
                    for pattern in dcm_patterns:
                        files = sorted(glob.glob(pattern))
                        if files:
                            print(f"Found DICOM images with pattern {pattern} in {subdir2}: {files}")
                            dcm_files.extend(files)
            
            self.image_paths.append(dcm_files)
            print(f"Total DICOM files found in {pdir}: {dcm_files}")

        if not self.patient_dirs or all(len(paths) == 0 for paths in self.image_paths):
            raise FileNotFoundError(f"No DICOM files found in {root_dir}. Check dataset structure, permissions, and file extensions. Found patient dirs: {self.patient_dirs}, Found DICOM files: {self.image_paths}")

        # Split patients (assuming balanced split for simplicity)
        total_patients = len(self.image_paths)
        if split == "train":
            self.image_paths = self.image_paths[:int(0.7 * total_patients)]
        elif split == "validation":
            self.image_paths = self.image_paths[int(0.7 * total_patients):int(0.85 * total_patients)]
        elif split == "test":
            self.image_paths = self.image_paths[int(0.85 * total_patients):]

        if self.preload_data:
            self.preloaded_labels = [self.load_label(i) for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.load_volume(index)
        label = self.load_label(index)

        # Ensure image is a NumPy array before applying transform
        if not isinstance(image, np.ndarray):
            image = image.numpy()  # Convert tensor back to NumPy if needed

        if self.transform:
            image = self.transform(image)
            # Ensure label has the correct shape for 3D interpolation (N, C, D, H, W) to match image
            if label.dim() == 3:  # Shape [D, H, W] -> [1, 1, D, H, W]
                label = label.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif label.dim() == 4:  # Shape [1, D, H, W] -> Ensure correct format
                label = label.unsqueeze(0)  # Add channel if missing
            # Interpolate to match image dimensions [160, 160, 96] (D, H, W)
            label = F.interpolate(label.float(), size=(160, 160, 96), mode='nearest').long()

        # Debug: Print shapes
        print(f"Image shape after transform: {image.shape}")
        print(f"Label shape before interpolation: {label.shape}")
        print(f"Label shape after interpolation: {label.shape}")

        return image, label

    def load_volume(self, index):
        """Loads a 3D volume by stacking DICOM slices."""
        slice_paths = self.image_paths[index]
        target_depth = 96  # Match the depth used in the model
        slices = [self.load_slice(p) for p in slice_paths[:target_depth]]
        if len(slices) < target_depth:
            slices.extend([np.zeros_like(slices[0])] * (target_depth - len(slices)))
        elif len(slices) > target_depth:
            slices = slices[:target_depth]
        image = np.stack(slices, axis=0)  # [96, 512, 512]
        image = np.expand_dims(image, axis=0)  # [1, 96, 512, 512]
        return image

    def load_slice(self, path):
        """Loads and preprocesses a single DICOM slice."""
        dicom_data = pydicom.dcmread(path)
        image = dicom_data.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image

    def load_label(self, index):
        """Placeholder for 3D label; replace with real data."""
        label = self.get_label_from_path(self.image_paths[index][0])
        if isinstance(label, (int, float)):
            label = np.full((96, 512, 512), label, dtype=np.uint8)  # 3D placeholder [D, H, W]
        elif isinstance(label, np.ndarray):
            if label.shape != (96, 512, 512):
                raise ValueError(f"Unexpected label shape {label.shape}, expected (96, 512, 512).")
        label = torch.tensor(label, dtype=torch.long)  # Shape [96, 512, 512]
        return label

    def get_label_from_path(self, path):
        """Placeholder for label logic; replace with actual label retrieval."""
        return 1  # Replace with real label logic (e.g., .nii files or segmented DICOMs)