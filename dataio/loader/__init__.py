import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.new_pancreas_ds import PancreasDataset

def get_dataset(name):
    """Get dataset class based on name."""
    datasets = {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax': HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'pancreas': PancreasDataset  # Added PancreasDataset
    }
    
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(datasets.keys())}")
    
    return datasets[name]

def get_dataset_path(dataset_name, opts):
    """Get dataset path from options."""
    return getattr(opts, dataset_name, None)
