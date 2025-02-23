import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
import time

from dataio.loader import PancreasDataset
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
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

    error_logger = ErrorLogger()

    model = get_model(json_opts.model)
    model.set_scheduler(train_opts)

    # Initialize data for tabulation
    training_data = {
        'Epoch': [],
        'Train Seg_Loss': [],
        'Validation Seg_Loss': [],
        'Test Seg_Loss': [],
        'Training Time (s)': [],
        'Validation Time (s)': [],
        'Test Time (s)': []
    }

    # Train for multiple epochs
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print(f'(Epoch {epoch}, Total Iterations: {len(train_loader)})')

        # Train
        model.net.train()  # Set model to training mode
        train_start_time = time.time()  # Start timing for training
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            model.set_input(images, labels)
            model.optimize_parameters()

            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        train_time = time.time() - train_start_time  # End timing for training

        # Validate
        model.net.eval()  # Set model to evaluation mode
        val_start_time = time.time()  # Start timing for validation
        for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
            model.set_input(images, labels)
            model.validate()
            errors = model.get_current_errors()
            stats = model.get_segmentation_stats()
            error_logger.update({**errors, **stats}, split='validation')
        val_time = time.time() - val_start_time  # End timing for validation

        # Test
        test_start_time = time.time()  # Start timing for testing
        for epoch_iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
            model.set_input(images, labels)
            model.validate()
            errors = model.get_current_errors()
            stats = model.get_segmentation_stats()
            error_logger.update({**errors, **stats}, split='test')
        test_time = time.time() - test_start_time  # End timing for testing

        # Calculate average losses for the epoch
        train_loss = np.mean(error_logger.get_errors('train').get('Seg_Loss', [0])) if error_logger.get_errors('train').get('Seg_Loss', []) else 0
        val_loss = np.mean(error_logger.get_errors('validation').get('Seg_Loss', [0])) if error_logger.get_errors('validation').get('Seg_Loss', []) else 0
        test_loss = np.mean(error_logger.get_errors('test').get('Seg_Loss', [0])) if error_logger.get_errors('test').get('Seg_Loss', []) else 0

        # Store data for tabulation
        training_data['Epoch'].append(epoch)
        training_data['Train Seg_Loss'].append(train_loss)
        training_data['Validation Seg_Loss'].append(val_loss)
        training_data['Test Seg_Loss'].append(test_loss)
        training_data['Training Time (s)'].append(train_time)
        training_data['Validation Time (s)'].append(val_time)
        training_data['Test Time (s)'].append(test_time)

        # Print errors for each split (replacing visualizer.print_current_errors)
        for split in ['train', 'validation', 'test']:
            errors = error_logger.get_errors(split)
            print(f"(epoch: {epoch}, split: {split}) ", end='')
            for key, value in errors.items():
                print(f"{key}: {np.mean(value) if value else 0:.3f} ", end='')
            print()

        error_logger.reset()

        # Save model checkpoint
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update learning rate
        model.update_learning_rate()

    # Create and display/save the table after training
    df = pd.DataFrame(training_data)
    if not df.empty:  # Only print and save if there’s data
        print("\nTraining Summary Table:")
        print(df.to_string(index=False))  # Print to console
        df.to_csv('training_summary.csv', index=False)  # Save to a CSV file in the current directory
    else:
        print("\nNo training data collected for the summary table. Check loss logging in error_logger or model.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Segmentation Training')
    parser.add_argument('-c', '--config', help='Training config file', required=True)
    parser.add_argument('-d', '--debug', help='Debug mode', action='store_true')
    args = parser.parse_args()

    train(args)