import os
import random
import numpy as np
import torch
import logging
import csv
from pathlib import Path

def set_all_random_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class FrameSampler:
    """Frame sampling class that supports augmentation and different sampling methods."""
    
    def __init__(self, sampling_method='uniform', num_frames=8, seed=42, 
                 logger=None, csv_save_dir=None, augment=False, max_aug_rounds=None):
        """
        Initialize the frame sampler.
        
        Args:
            sampling_method (str): 'uniform', 'random', or 'random_window'
            num_frames (int): Number of frames to sample
            seed (int): Random seed for reproducibility
            logger (logging.Logger): Logger instance
            csv_save_dir (str): Directory to save sampling CSV files
            augment (bool): Whether to use data augmentation
            max_aug_rounds (int, optional): Maximum number of augmentation rounds
        """
        self.sampling_method = sampling_method
        self.num_frames = num_frames
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.csv_save_dir = csv_save_dir
        self.augment = augment
        self.max_aug_rounds = max_aug_rounds
        
        # For caching sampled indices
        self.cached_indices = {}
        
        # For augmentation
        self.aug_video_map = []  # Maps augmented index to (video_idx, aug_round)
        self.augmented_labels = []
        self.original_length = 0
        
        # Set random seed for reproducibility
        self.set_random_seed(seed)
        
        self.logger.info(f"Initialized FrameSampler with {sampling_method} sampling, {num_frames} frames")
        if self.augment:
            self.logger.info(f"Data augmentation enabled, max rounds: {max_aug_rounds or 'auto'}")
    
    def set_random_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def setup_augmentation(self, video_paths, labels, total_frames_func):
        """
        Initialize augmentation information for all videos in the dataset.
        
        Args:
            video_paths: List of paths to all videos
            labels: List of labels for all videos
            total_frames_func: Function to get total frames for a video path
        """
        if not self.augment:
            return
            
        self.logger.info(f"Setting up data augmentation with {self.sampling_method} sampling method")
        
        # Store original dataset length before augmentation
        self.original_length = len(video_paths)
        
        # Reset mappings for augmented samples
        self.aug_video_map = []
        self.augmented_labels = []
        
        # Count samples for each class before augmentation
        class_counts_before = {}
        for label in labels:
            class_counts_before[label] = class_counts_before.get(label, 0) + 1
        
        # Process each video to calculate augmentation rounds and store mappings
        total_augmented_samples = 0
        
        for video_idx, video_path in enumerate(video_paths):
            # Get video frame count
            total_frames = total_frames_func(video_path)
            
            # Calculate max augmentation rounds for this video
            if self.max_aug_rounds is None:
                video_max_rounds = self.calculate_max_aug_rounds(total_frames, self.num_frames)
            else:
                video_max_rounds = min(self.max_aug_rounds, 
                                     self.calculate_max_aug_rounds(total_frames, self.num_frames))
            
            # Get the video's label
            label = labels[video_idx]
            
            # Add mapping entries for augmented samples
            for aug_round in range(1, video_max_rounds + 1):
                self.aug_video_map.append((video_idx, aug_round))
                self.augmented_labels.append(label)
                total_augmented_samples += 1
        
        # Calculate class counts after augmentation
        class_counts_after = class_counts_before.copy()
        for label in self.augmented_labels:
            class_counts_after[label] = class_counts_after.get(label, 0) + 1
        
        # Log augmentation info
        self.logger.info(f"Data augmentation added {total_augmented_samples} samples to the original {self.original_length}")
        
        # Create class name mapping for nicer output
        class_names = {0: "non-referral", 1: "referral"}
        
        # Calculate and log class distribution before augmentation
        original_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_before.items())])
        self.logger.info(f"Original class distribution: {original_dist_str}")
        
        # Calculate and log class distribution after augmentation
        augmented_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_after.items())])
        self.logger.info(f"Augmented class distribution: {augmented_dist_str}")
        
        # Calculate added samples per class
        added_counts = {k: class_counts_after.get(k, 0) - class_counts_before.get(k, 0) for k in set(class_counts_before) | set(class_counts_after)}
        added_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: +{v}" for k, v in sorted(added_counts.items())])
        self.logger.info(f"Added samples per class: {added_dist_str}")
        
        # Calculate class distribution percentages
        original_total = sum(class_counts_before.values())
        original_pct = {k: (v / original_total) * 100 for k, v in class_counts_before.items()}
        original_pct_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.1f}%" for k, v in sorted(original_pct.items())])
        
        augmented_total = sum(class_counts_after.values())
        augmented_pct = {k: (v / augmented_total) * 100 for k, v in class_counts_after.items()}
        augmented_pct_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.1f}%" for k, v in sorted(augmented_pct.items())])
        
        self.logger.info(f"Original class percentages: {original_pct_str}")
        self.logger.info(f"Augmented class percentages: {augmented_pct_str}")
        
        # Calculate augmentation factor per class
        aug_factor = {k: class_counts_after.get(k, 0) / class_counts_before.get(k, 1) for k in class_counts_before}
        aug_factor_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.2f}x" for k, v in sorted(aug_factor.items())])
        self.logger.info(f"Augmentation factor per class: {aug_factor_str}")
    
    def calculate_max_aug_rounds(self, total_frames, num_frames):
        """
        Calculate the maximum number of augmentation rounds based on uniform sampling.
        
        Args:
            total_frames: Total number of frames in the video
            num_frames: Number of frames to sample per round
            
        Returns:
            max_rounds: Maximum number of augmentation rounds
        """
        # Edge case: if num_frames is too large relative to total_frames,
        # the chunks will be too small for meaningful augmentation
        if num_frames > total_frames / 2:
            # When requesting too many frames, limit augmentation rounds
            return max(1, min(5, total_frames // (2 * num_frames) + 1))
        
        # For uniform sampling with original method:
        if num_frames <= 1:
            return 1  # No augmentation possible with only 1 frame
        
        # Calculate step using original formula
        step = (total_frames - 1) / (num_frames - 1)
        
        # Find the minimum space between consecutive frames - this determines max rounds
        min_chunk_size = max(1, int(step) - 1)  # At least 1 frame between borders
        
        # Each augmentation round uses one frame from each chunk
        max_rounds = max(1, min_chunk_size)
        
        return max_rounds
    
    def get_dataset_size(self, original_size):
        """
        Return the total length of the dataset including augmentations.
        
        Args:
            original_size: Original size of the dataset
            
        Returns:
            int: Total size including augmentations
        """
        if self.augment:
            return original_size + len(self.aug_video_map)
        return original_size
    
    def get_item_info(self, idx, original_size):
        """
        Get information about the item at the given index.
        
        Args:
            idx: Index of the item
            original_size: Original size of the dataset
            
        Returns:
            dict: Information about video and augmentation
        """
        # Handle augmented indices
        aug_round = None
        if self.augment and idx >= original_size:
            # This is an augmented sample
            aug_idx = idx - original_size
            video_idx, aug_round = self.aug_video_map[aug_idx]
            return {
                'original_idx': video_idx,
                'aug_round': aug_round
            }
        else:
            # Regular sample from original dataset
            return {
                'original_idx': idx,
                'aug_round': None
            }
    
    def get_frame_indices(self, video_path, total_frames, aug_round=None):
        """
        Get frame indices based on sampling method with augmentation support.
        
        Args:
            video_path: Path to the video file
            total_frames: Total number of frames in the video
            aug_round: Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices to sample
        """
        # Generate a cache key that includes augmentation round
        cache_key = f"{video_path}_{aug_round}"
        
        # Check if we already have cached indices for this video and aug round
        if cache_key in self.cached_indices:
            return self.cached_indices[cache_key]
        
        # Set a video-specific seed based on the filename for consistent sampling
        video_seed = int(hash(os.path.basename(video_path)) % 10000000)
        if aug_round is not None:
            # Add augmentation round to seed for different but reproducible augmentations
            video_seed = video_seed + aug_round * 1000
            
        random.seed(video_seed)
        np.random.seed(video_seed)
        
        # Handle different sampling methods with augmentation support
        if self.sampling_method == 'uniform':
            # UNIFORM SAMPLING with augmentation
            indices = self._get_uniform_sampling_indices(total_frames, aug_round)
        elif self.sampling_method == 'random':
            # RANDOM SAMPLING with augmentation
            indices = self._get_random_sampling_indices(total_frames, aug_round)
# Continuing from where we left off in utils/custom_sampling.py

        else:  # 'random_window'
            # RANDOM WINDOW SAMPLING with augmentation
            indices = self._get_random_window_sampling_indices(total_frames, aug_round)
            
        # Reset the global random seed
        self.set_random_seed(self.seed)
        
        # Cache the indices
        self.cached_indices[cache_key] = indices
        
        return indices
    
    def _get_uniform_sampling_indices(self, total_frames, aug_round):
        """
        Implement uniform sampling with augmentation support.
        
        Args:
            total_frames (int): Total frames in the video
            aug_round (int, optional): Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # Calculate border positions first (these are the same with or without augmentation)
        if self.num_frames == 1:
            border_indices = [total_frames // 2]  # Middle frame for single frame
        else:
            step = (total_frames - 1) / (self.num_frames - 1)
            border_indices = [min(int(i * step), total_frames - 1) for i in range(self.num_frames)]
        
        # Original sampling (no augmentation) just returns the borders
        if aug_round is None:
            return border_indices
        
        # For augmentation, we need to sample frames from each chunk
        num_chunks = len(border_indices) - 1
        round_frames = []
        
        # For each chunk, select one frame based on the augmentation round
        for i in range(num_chunks):
            chunk_start = border_indices[i]     # Left border
            chunk_end = border_indices[i + 1]   # Right border
            
            # Calculate frame index for this round
            frame_idx = chunk_start + aug_round
            
            # Ensure we stay within the chunk (excluding right border)
            if frame_idx < chunk_end:
                round_frames.append(frame_idx)
            else:
                # If we run out of frames, use repetition within chunk
                available_frames = list(range(chunk_start + 1, chunk_end))
                if available_frames:
                    # Create deterministic choice from available frames
                    frame_idx = available_frames[hash(i + aug_round) % len(available_frames)]
                    round_frames.append(frame_idx)
                else:
                    # If no frames available, use the left border again
                    round_frames.append(chunk_start)
        
        # Always include the last border for completeness
        if border_indices:
            round_frames.append(border_indices[-1])
            
        return sorted(round_frames)
    
    def _get_random_sampling_indices(self, total_frames, aug_round):
        """
        Implement random sampling with augmentation support.
        
        Args:
            total_frames (int): Total frames in the video
            aug_round (int, optional): Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # For both original and augmentation rounds, random sampling is similar,
        # but we use a different seed for each augmentation round
        if total_frames >= self.num_frames:
            # Random sampling without replacement when we have enough frames
            indices = sorted(random.sample(range(total_frames), self.num_frames))
        else:
            # Random sampling with replacement when we don't have enough frames
            indices = sorted(random.choices(range(total_frames), k=self.num_frames))
            
        return indices
    
    def _get_random_window_sampling_indices(self, total_frames, aug_round):
        """
        Implement random window sampling with augmentation support.
        
        Args:
            total_frames (int): Total frames in the video
            aug_round (int, optional): Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # Calculate window size
        window_size = total_frames / self.num_frames
        indices = []
        
        # For each window, select one random frame
        for i in range(self.num_frames):
            start = int(i * window_size)
            end = min(int((i + 1) * window_size), total_frames)
            end = max(end, start + 1)  # Ensure window has at least 1 frame
            
            # For original sampling or if we only have one frame in the window
            if aug_round is None or start == end - 1:
                frame_idx = random.randint(start, end - 1)
            else:
                # For augmentation, try to select different frames in each round
                # Create a hash of window+round for deterministic but varied selection
                window_round_hash = (i * 1000 + aug_round) % (end - start)
                frame_idx = start + window_round_hash % (end - start)
                
            indices.append(frame_idx)
            
        return sorted(indices)
    
    def save_sampling_indices(self, dataset_name, split, video_paths, labels, total_frames_func):
        """
        Save the sampled indices to a CSV file for reproducibility.
        
        Args:
            dataset_name (str): Name of the dataset
            split (str): 'train', 'val', or 'test'
            video_paths (list): List of video paths
            labels (list): List of labels
            total_frames_func (function): Function to get total frames for a video
        """
        if not self.csv_save_dir:
            self.logger.warning("No CSV save directory provided, cannot save sampled indices")
            return
            
        # Create CSV filename based on dataset, split, and sampling method
        csv_file = os.path.join(self.csv_save_dir, 
                               f"sampling_indices_{dataset_name}_{self.sampling_method}_{split}.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # Write to CSV
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['video_filename', 'total_frames', 'sampled_frames', 'aug_round', 'label'])
            
            # Write data for each video
            for i, video_path in enumerate(sorted(video_paths)):
                # Get video frame count
                total_frames = total_frames_func(video_path)
                
                # Get the video's label
                label = labels[i]
                
                # First, write the original sampling
                indices = self.get_frame_indices(video_path, total_frames)
                indices_str = ','.join(map(str, indices))
                video_filename = os.path.basename(video_path)
                writer.writerow([video_filename, total_frames, indices_str, 'original', label])
                
                # Then write each augmentation round if augmentation is enabled
                if self.augment:
                    # Calculate max rounds for this video
                    max_rounds = self.calculate_max_aug_rounds(total_frames, self.num_frames)
                    if self.max_aug_rounds is not None:
                        max_rounds = min(max_rounds, self.max_aug_rounds)
                    
                    # Write entries for each augmentation round
                    for aug_round in range(1, max_rounds + 1):
                        # Get the frame indices for this augmentation round
                        indices = self.get_frame_indices(video_path, total_frames, aug_round)
                        indices_str = ','.join(map(str, indices))
                        video_filename = os.path.basename(video_path)
                        writer.writerow([video_filename, total_frames, indices_str, f'aug_{aug_round}', label])
        
        self.logger.info(f"Saved sampled frame records to {csv_file}")