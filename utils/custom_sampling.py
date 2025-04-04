import random
import numpy as np
import torch
import os
import csv
from pathlib import Path
import logging

def set_all_random_seeds(seed):
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed (int): Seed value for random number generators
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set hash seed for Python processes
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_sampling_indices(total_frames, num_frames, sampling_method, seed=42):
    """
    Get frame indices based on the specified sampling method.
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        sampling_method (str): 'uniform', 'random', or 'random_window'
        seed (int, optional): Random seed for reproducibility (default: 42)
        
    Returns:
        list: Selected frame indices
    """
    # Store original random states
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For videos with enough frames, use standard sampling
    if total_frames >= num_frames:
        if sampling_method == 'random':
            # Random sampling without replacement
            indices = sorted(random.sample(range(total_frames), num_frames))
        elif sampling_method == 'random_window':
            # Random window sampling
            window_size = total_frames / num_frames
            indices = []
            for i in range(num_frames):
                start = int(i * window_size)
                end = min(int((i + 1) * window_size), total_frames)
                end = max(end, start + 1)  # Ensure window has at least 1 frame
                frame_idx = random.randint(start, end - 1)
                indices.append(frame_idx)
        else:  # Default to uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                step = (total_frames - 1) / (num_frames - 1)
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    # For videos with fewer frames than requested, use dynamic sampling
    else:
        if sampling_method == 'random':
            # With dynamic sampling, we'll need to allow duplicates
            indices = sorted(random.choices(range(total_frames), k=num_frames))
        elif sampling_method == 'random_window':
            # For random window with fewer frames, create virtual windows smaller than 1 frame
            indices = []
            window_size = total_frames / num_frames  # Will be < 1
            
            for i in range(num_frames):
                # Calculate virtual window boundaries
                virtual_start = i * window_size
                virtual_end = (i + 1) * window_size
                
                # Convert to actual frame indices with potential duplicates
                actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                  total_frames - 1)
                indices.append(actual_index)
        else:  # Uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                # Create evenly spaced indices that might include duplicates
                step = total_frames / num_frames
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    # Restore original random states to prevent affecting other code
    random.setstate(random_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    
    return indices

class FrameSampler:
    """
    A class for sampling frames from videos using different strategies.
    This can be used to extend existing dataset classes.
    """
    def __init__(self, csv_save_dir=None, logger=None):
        """
        Initialize the FrameSampler.
        
        Args:
            csv_save_dir (str): Directory to save CSV files with sampling indices
            logger: Logger instance for logging
        """
        self.csv_save_dir = csv_save_dir
        self.logger = logger or logging.getLogger(__name__)
        self.sampled_indices = {}  # Dict to store sampled indices for each dataset/method
        
        # Create directories for saving CSVs if specified
        if csv_save_dir:
            os.makedirs(csv_save_dir, exist_ok=True)
            
        # Set seed for reproducibility
        set_all_random_seeds(42)
    
    def sample_frames(self, video_path, total_frames, num_frames, sampling_method, dataset_name='unknown', seed=42):
        """
        Sample frame indices from a video and store for later export.
        
        Args:
            video_path: Path to the video file
            total_frames: Total number of frames in the video
            num_frames: Number of frames to sample
            sampling_method: 'uniform', 'random', or 'random_window'
            dataset_name: Name of the dataset (for organizing CSV files)
            seed: Random seed for reproducibility (default: 42)
            
        Returns:
            list: Selected frame indices
        """
        # Generate a deterministic video-specific seed based on filename
        video_seed = seed
        if seed is None:
            # Use default seed of 42 for reproducibility
            video_seed = 42
        
        # Get sampling indices with fixed seed
        indices = get_sampling_indices(total_frames, num_frames, sampling_method, video_seed)
        
        # Store the indices
        key = f"{dataset_name}_{sampling_method}"
        if key not in self.sampled_indices:
            self.sampled_indices[key] = []
            
        # Extract just the filename from the video path for cleaner output
        filename = os.path.basename(video_path)
        
        # Store the sampling information
        self.sampled_indices[key].append({
            'video_path': filename,
            'total_frames': total_frames,
            'indices': indices,
            'sampling_method': sampling_method,
            'seed': video_seed
        })
        
        return indices
    
    def save_all_indices_to_csv(self):
        """
        Save all collected sampling indices to CSV files.
        Each dataset and sampling method combination gets its own CSV file.
        """
        if not self.csv_save_dir:
            self.logger.warning("No CSV save directory provided, cannot save sampling indices")
            return
            
        for key, indices_list in self.sampled_indices.items():
            csv_path = os.path.join(self.csv_save_dir, f"sampling_indices_{key}.csv")
            
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(['video_filename', 'total_frames', 'sampled_indices', 'sampling_method', 'seed'])
                    
                    # Write data for each video
                    for item in indices_list:
                        indices_str = ','.join(map(str, item['indices']))
                        writer.writerow([
                            item['video_path'],
                            item['total_frames'],
                            indices_str,
                            item['sampling_method'],
                            item['seed']
                        ])
                
                self.logger.info(f"Saved {len(indices_list)} sampling records to {csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving sampling indices to CSV: {str(e)}")
    
    def save_indices_by_dataset(self, dataset_name, split='train'):
        """
        Save the sampling indices for a specific dataset to a CSV file.
        Modified to match the expected CSV format.
        
        Args:
            dataset_name (str): Name of the dataset
            split (str): Data split (train, val, test)
        """
        if not self.csv_save_dir:
            self.logger.warning("No CSV save directory provided, cannot save sampling indices")
            return
            
        for sampling_method in ['uniform', 'random', 'random_window']:
            key = f"{dataset_name}_{sampling_method}"
            if key not in self.sampled_indices:
                continue
                
            csv_path = os.path.join(self.csv_save_dir, f"sampling_indices_{key}_{split}.csv")
            
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header with expected format
                    writer.writerow(['video_filename', 'total_frames', 'sampled_frames'])
                    
                    # Write data for each video in expected format
                    for item in self.sampled_indices[key]:
                        # First two columns
                        row = [item['video_path'], item['total_frames']]
                        # Add all indices as additional columns
                        row.extend(item['indices'])
                        writer.writerow(row)
                
                self.logger.info(f"Saved {len(self.sampled_indices[key])} sampling records to {csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving sampling indices to CSV: {str(e)}")