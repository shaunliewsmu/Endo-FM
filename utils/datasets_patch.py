# monkey_patch.py
# This script applies the sampling tracking functionality directly to the dataset classes
# without requiring imports that might fail

import os
import logging
import csv
import random
import numpy as np
import torch

def patch_datasets():
    """Apply the patch directly to the dataset classes after they've been imported"""
    from datasets import UCF101, HMDB51, Kinetics
    
    # Print information about the modules to confirm they're found
    print(f"Found dataset classes: UCF101, HMDB51, Kinetics")
    print(f"UCF101 is located at: {UCF101.__module__}")
    
    # Store original methods to call later
    original_ucf_sample = UCF101._sample_indices
    original_hmdb_sample = HMDB51._sample_indices
    original_kinetics_sample = Kinetics._sample_indices
    
    # Create a FrameSampler class
    class FrameSampler:
        def __init__(self, csv_save_dir=None, logger=None):
            self.csv_save_dir = csv_save_dir
            self.logger = logger or logging.getLogger(__name__)
            self.sampled_indices = {}
            
            if csv_save_dir:
                os.makedirs(csv_save_dir, exist_ok=True)
                
        def sample_frames(self, video_path, total_frames, num_frames, sampling_method, dataset_name='unknown', seed=42):
            # Generate sampled indices
            indices = get_sampling_indices(total_frames, num_frames, sampling_method, seed)
            
            # Store the indices
            key = f"{dataset_name}_{sampling_method}"
            if key not in self.sampled_indices:
                self.sampled_indices[key] = []
                
            # Extract filename from path
            filename = os.path.basename(video_path)
            
            # Store sampling info
            self.sampled_indices[key].append({
                'video_path': filename,
                'total_frames': total_frames,
                'indices': indices,
                'sampling_method': sampling_method,
                'seed': seed
            })
            
            return indices
            
        def save_indices_by_dataset(self, dataset_name, split='train'):
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
                    print(f"Saved {len(self.sampled_indices[key])} sampling records to {csv_path}")
                except Exception as e:
                    self.logger.error(f"Error saving sampling indices to CSV: {str(e)}")
                    print(f"Error saving sampling indices to CSV: {str(e)}")
    
    # Add instance variables and methods directly to the classes
    def init_sampler(self, csv_save_dir, logger, dataset_name, split, sampling_method=None):
        self.frame_sampler = FrameSampler(csv_save_dir, logger)
        self.dataset_name = dataset_name
        self.split = split
        
        # Get the appropriate sampling method
        if sampling_method:
            self.sampling_method = sampling_method
        elif hasattr(self, 'cfg') and hasattr(self.cfg, 'DATA'):
            # Use the config if available
            if split == "train" and hasattr(self.cfg.DATA, 'TRAIN_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.TRAIN_SAMPLING_METHOD
            elif split == "val" and hasattr(self.cfg.DATA, 'VAL_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.VAL_SAMPLING_METHOD
            elif hasattr(self.cfg.DATA, 'TEST_SAMPLING_METHOD'):
                self.sampling_method = self.cfg.DATA.TEST_SAMPLING_METHOD
            else:
                self.sampling_method = 'uniform'  # Default
        else:
            self.sampling_method = 'uniform'  # Default
    
    def get_sampled_frames(self, video_path, total_frames, num_frames, sampling_method):
        return self.frame_sampler.sample_frames(
            video_path, 
            total_frames, 
            num_frames, 
            sampling_method,
            dataset_name=self.dataset_name,
            seed=42  # Fixed seed for reproducibility
        )
    
    def save_sampling_indices(self):
        if hasattr(self, 'frame_sampler'):
            self.frame_sampler.save_indices_by_dataset(self.dataset_name, self.split)
    
    # Override the sample_indices methods
    def ucf_sample_indices(self, num_frames):
        if hasattr(self, 'frame_sampler') and hasattr(self, 'sampling_method'):
            # Get total frames
            if self._video_meta[self._index]['num_frames'] <= 0:
                total_frames = 300  # Default
            else:
                total_frames = self._video_meta[self._index]['num_frames']
            
            # Use the sampler
            return get_sampled_frames(
                self,
                video_path=self._path_to_videos[self._index],
                total_frames=total_frames,
                num_frames=num_frames,
                sampling_method=self.sampling_method
            )
        else:
            # Fall back to original method
            return original_ucf_sample(self, num_frames)
    
    def hmdb_sample_indices(self, num_frames):
        if hasattr(self, 'frame_sampler') and hasattr(self, 'sampling_method'):
            # Get total frames
            if self._video_meta[self._index]['num_frames'] <= 0:
                total_frames = 300  # Default
            else:
                total_frames = self._video_meta[self._index]['num_frames']
            
            # Use the sampler
            return get_sampled_frames(
                self,
                video_path=self._path_to_videos[self._index],
                total_frames=total_frames,
                num_frames=num_frames,
                sampling_method=self.sampling_method
            )
        else:
            # Fall back to original method
            return original_hmdb_sample(self, num_frames)
    
    def kinetics_sample_indices(self, num_frames):
        if hasattr(self, 'frame_sampler') and hasattr(self, 'sampling_method'):
            # Get total frames
            if self._video_meta[self._index]['num_frames'] <= 0:
                total_frames = 300  # Default
            else:
                total_frames = self._video_meta[self._index]['num_frames']
            
            # Use the sampler
            return get_sampled_frames(
                self,
                video_path=self._path_to_videos[self._index],
                total_frames=total_frames,
                num_frames=num_frames,
                sampling_method=self.sampling_method
            )
        else:
            # Fall back to original method
            return original_kinetics_sample(self, num_frames)
    
    # Add all methods to the classes
    UCF101.init_sampler = init_sampler
    HMDB51.init_sampler = init_sampler
    Kinetics.init_sampler = init_sampler
    
    UCF101.get_sampled_frames = get_sampled_frames
    HMDB51.get_sampled_frames = get_sampled_frames
    Kinetics.get_sampled_frames = get_sampled_frames
    
    UCF101.save_sampling_indices = save_sampling_indices
    HMDB51.save_sampling_indices = save_sampling_indices
    Kinetics.save_sampling_indices = save_sampling_indices
    
    # Replace _sample_indices methods
    UCF101._sample_indices = ucf_sample_indices
    HMDB51._sample_indices = hmdb_sample_indices
    Kinetics._sample_indices = kinetics_sample_indices
    
    print("Successfully patched dataset classes with sampling functionality")

# Reuse the sampling functions from your custom_sampling.py
def set_all_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_sampling_indices(total_frames, num_frames, sampling_method, seed=42):
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

if __name__ == "__main__":
    # This will be called directly in the training script
    pass