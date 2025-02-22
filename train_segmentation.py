import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loader import PancreasDataset
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models import get_model

def train(arguments):
    json_filename = arguments.config
    network_debug = arguments.debug

    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    arch_type = json_opts.model.model_type  # 'unet_ct_dsv'

    # Use Pancreas_Small1 path instead of config's path
    dataset_path = "/fab3/btech/2022/sreejita.das22b/Attention-Gated-Networks/Pancreas_Small1"

    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    train_dataset = PancreasDataset(dataset_path, split='train', transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = PancreasDataset(dataset_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset = PancreasDataset(dataset_path, split='test', transform=ds_transform['valid'], preload_data=train_opts.preloadData)

    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=False)

    print("Debugging DataLoader output:")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: Images shape {images.shape}, Labels shape {labels.shape}")
        break

    visualizer = Visualiser(json_opts.visualisation, save_dir="model_outputs/")
    error_logger = ErrorLogger()

    model = get_model(json_opts.model)
    model.set_scheduler(train_opts)

    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print(f'(Epoch {epoch}, Total Iterations: {len(train_loader)})')

        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            model.set_input(images, labels)
            model.optimize_parameters()

            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):
                model.set_input(images, labels)
                model.validate()
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        model.update_learning_rate()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Segmentation Training')
    parser.add_argument('-c', '--config', help='Training config file', required=True)
    parser.add_argument('-d', '--debug', help='Debug mode', action='store_true')
    args = parser.parse_args()

    train(args)