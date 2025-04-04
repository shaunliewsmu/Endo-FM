# Update to visualize_sampling.py to handle different CSV formats
# and to use the correct sampling method from the training script

import pandas as pd
import os
import glob
import logging

def find_existing_csv_files(output_dir):
    """
    Find all existing CSV files in the output directory.
    Returns a list of (dataset, sampling_method, split) tuples.
    """
    csv_dir = os.path.join(output_dir, 'sampling_indices')
    if not os.path.exists(csv_dir):
        return []
        
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    result = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith('sampling_indices_'):
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 5:
                # Format: sampling_indices_dataset_method_split.csv
                dataset = parts[2]
                method = parts[3]
                split = parts[4]
                result.append((dataset, method, split, csv_file))
    
    return result

def get_csv_format(csv_path):
    """
    Determine the format of the CSV file by reading the header.
    Returns either 'standard' or 'columnar'.
    
    'standard' format:
    video_filename,total_frames,sampled_frames,...
    
    'columnar' format:
    video_filename,total_frames,sampled_indices,sampling_method,seed
    """
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            if 'sampled_frames' in header:
                return 'standard'
            elif 'sampled_indices' in header:
                return 'columnar'
            else:
                return 'unknown'
    except:
        return 'unknown'

def read_sampling_indices(csv_path):
    """
    Read sampling indices from a CSV file, handling different formats.
    Returns a list of dictionaries with 'video_path', 'total_frames', 'indices'.
    """
    format_type = get_csv_format(csv_path)
    result = []
    
    try:
        df = pd.read_csv(csv_path)
        
        if format_type == 'standard':
            # Standard format: video_filename,total_frames,sampled_frames,...
            for _, row in df.iterrows():
                indices = row.values[2:].tolist()  # All columns from the 3rd onwards
                indices = [int(idx) for idx in indices if not pd.isna(idx)]
                result.append({
                    'video_path': row['video_filename'],
                    'total_frames': int(row['total_frames']),
                    'indices': indices
                })
        
        elif format_type == 'columnar':
            # Columnar format: video_filename,total_frames,sampled_indices,sampling_method,seed
            for _, row in df.iterrows():
                indices = [int(idx) for idx in row['sampled_indices'].split(',')]
                result.append({
                    'video_path': row['video_filename'],
                    'total_frames': int(row['total_frames']),
                    'indices': indices
                })
        
        else:
            # Try to guess the format
            for _, row in df.iterrows():
                if 'sampled_indices' in df.columns:
                    indices = [int(idx) for idx in row['sampled_indices'].split(',')]
                else:
                    # Assume all columns from the 3rd onwards are indices
                    indices = row.values[2:].tolist()
                    indices = [int(idx) for idx in indices if not pd.isna(idx)]
                
                result.append({
                    'video_path': row['video_filename'] if 'video_filename' in df.columns else str(row.values[0]),
                    'total_frames': int(row['total_frames']) if 'total_frames' in df.columns else int(row.values[1]),
                    'indices': indices
                })
    
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {str(e)}")
    
    return result

# This function can be used to prepare data for visualization from any CSV format
def prepare_visualization_data(output_dir, dataset, train_sampling, val_sampling, test_sampling):
    """
    Prepare visualization data for all available sampling methods.
    
    Args:
        output_dir (str): Output directory where sampling indices are saved
        dataset (str): Dataset name (e.g., 'ucf101')
        train_sampling, val_sampling, test_sampling (str): Sampling methods from args
        
    Returns:
        dict: Mapping of (split, method) to list of sample data
    """
    csv_files = find_existing_csv_files(output_dir)
    visualization_data = {}
    
    # First, look for exact matches based on specified sampling methods
    for dataset_name, method, split, csv_path in csv_files:
        if dataset_name == dataset:
            if (split == 'train' and method == train_sampling) or \
               (split == 'val' and method == val_sampling) or \
               (split == 'test' and method == test_sampling):
                key = (split, method)
                visualization_data[key] = read_sampling_indices(csv_path)
    
    # If no exact matches, use any available sampling methods
    if not visualization_data:
        for dataset_name, method, split, csv_path in csv_files:
            if dataset_name == dataset:
                key = (split, method)
                visualization_data[key] = read_sampling_indices(csv_path)
    
    return visualization_data

# This is a standalone script that can be used to visualize sampling indices
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize sampling patterns')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing sampling indices')
    parser.add_argument('--dataset', type=str, default='ucf101', help='Dataset name')
    parser.add_argument('--train_sampling', type=str, default='random', help='Training sampling method')
    parser.add_argument('--val_sampling', type=str, default='uniform', help='Validation sampling method')
    parser.add_argument('--test_sampling', type=str, default='uniform', help='Testing sampling method')
    
    args = parser.parse_args()
    
    # Prepare visualization data
    data = prepare_visualization_data(
        args.output_dir, 
        args.dataset, 
        args.train_sampling, 
        args.val_sampling, 
        args.test_sampling
    )
    
    # Print summary
    for (split, method), samples in data.items():
        print(f"Found {len(samples)} samples for {split} split using {method} sampling method")