import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pprint import pprint

class Transformations:
    def __init__(self, name):
        self.name = name
        self.scale_size = (192, 192, 1)
        self.patch_size = (160, 160, 96)  # Updated to match label dimensions
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0
        self.division_factor = (16, 16, 1)

    def get_transformation(self):
        transform_dict = {
            'ukbb_sax': self.cmr_3d_sax_transform,
            'hms_sax': self.hms_sax_transform,
            'test_sax': self.test_3d_sax_transform,
            'acdc_sax': self.cmr_3d_sax_transform,
            'us': self.ultrasound_transform,
            'unet_ct_dsv': self.unet_3d_transform,
        }
        return transform_dict[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name, opts)
        if hasattr(t_opts, 'scale_size'): self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'): self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'): self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'): self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'): self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'): self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'): self.division_factor = t_opts.division_factor

    def unet_3d_transform(self):
        class Resize3D:
            def __init__(self, size):
                self.size = size

            def __call__(self, img):
                # Ensure img is NumPy array or tensor
                if isinstance(img, np.ndarray):
                    img_tensor = torch.from_numpy(img).float()
                else:
                    img_tensor = img.float()  # Handle if it's already a tensor
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.size, mode='trilinear', align_corners=False)
                return img_tensor.squeeze(0)

        train_transform = transforms.Compose([
            Resize3D(self.patch_size),  # [160, 160, 96]
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        valid_transform = transforms.Compose([
            Resize3D(self.patch_size),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def cmr_3d_sax_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.scale_size[:2]),
            transforms.RandomHorizontalFlip(p=self.random_flip_prob) if self.random_flip_prob > 0 else None,
            transforms.RandomAffine(degrees=self.rotate_val, translate=self.shift_val, scale=self.scale_val),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomCrop(self.patch_size[:2]),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.scale_size[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.CenterCrop(self.patch_size[:2]),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def test_3d_sax_transform(self):
        test_transform = transforms.Compose([
            transforms.Resize(self.scale_size[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        return {'test': test_transform}

    def ultrasound_transform(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=self.random_flip_prob) if self.random_flip_prob > 0 else None,
            transforms.RandomAffine(degrees=self.rotate_val, translate=self.shift_val, scale=self.scale_val),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        return {'train': train_transform, 'valid': valid_transform}

    def hms_sax_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.scale_size[:2]),
            transforms.RandomHorizontalFlip(p=self.random_flip_prob) if self.random_flip_prob > 0 else None,
            transforms.RandomAffine(degrees=self.rotate_val, translate=self.shift_val, scale=self.scale_val),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomCrop(self.patch_size[:2]),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.scale_size[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.CenterCrop(self.patch_size[:2]),
        ])

        return {'train': train_transform, 'valid': valid_transform}