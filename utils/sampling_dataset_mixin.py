# Updated utils/sampling_dataset_mixin.py

import logging
import random
import numpy as np
import torch
import os
import csv
from pathlib import Path

class SamplingDatasetMixin:
    """Mixin class to add sampling index tracking and augmentation to existing dataset classes."""
    
    def init_sampler(self, csv_save_dir, logger, dataset_name, split, sampling_method=None, 
                     augment=False, max_aug_rounds=None, aug_step_size=1):
        """Initialize the sampler for tracking sampling indices with augmentation support."""
        self.csv_save_dir = csv_save_dir
        self.logger = logger or logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.split = split
        
        # Augmentation settings - only enable for training
        self.augment = augment and split == 'train'
        self.max_aug_rounds = max_aug_rounds
        self.aug_step_size = aug_step_size
        
        # Cache for sampled indices
        self.cached_indices = {}
        
        # Store original dataset properties FIRST before any operations
        self.original_video_paths = self._path_to_videos.copy()
        self.original_labels = self._labels.copy()
        self.original_length = len(self.original_video_paths)
        
        # For mapping augmented indices to original video indices and aug rounds
        self.aug_video_map = []
        self.augmented_labels = []
        
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
        
        self.logger.info(f"Initialized {split} dataset with {self.original_length} videos, "
                        f"sampling method: {self.sampling_method}")
        
        # Set up augmentation AFTER setting all properties
        if self.augment:
            self.logger.info(f"Setting up augmentation for {split} dataset")
            self.setup_augmentation()
    
    def setup_augmentation(self):
        """Initialize augmentation for training data."""
        if not self.augment:
            self.logger.info("Augmentation is not enabled, skipping setup")
            return
        
        self.logger.info(f"Setting up data augmentation with {self.sampling_method} sampling method "
                       f"and step size {self.aug_step_size}")
        
        # Clear any existing augmentation mappings
        self.aug_video_map = []
        self.augmented_labels = []
        
        # Process each video to calculate augmentation rounds and create mappings
        total_augmented_samples = 0
        videos_processed = 0
        videos_with_augmentation = 0
        
        # Count samples for each class before augmentation
        class_counts_before = {}
        for label in self.original_labels:
            class_counts_before[label] = class_counts_before.get(label, 0) + 1
        
        for video_idx, video_path in enumerate(self.original_video_paths):
            try:
                videos_processed += 1
                
                # Get video frame count
                import cv2
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.logger.warning(f"Could not open video for augmentation: {video_path}")
                    continue
                    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Calculate max augmentation rounds for this video
                video_max_rounds = self.calculate_max_aug_rounds(total_frames)
                
                # Apply max rounds limit if specified
                if self.max_aug_rounds is not None:
                    video_max_rounds = min(video_max_rounds, self.max_aug_rounds)
                
                # Make sure we have at least one round
                video_max_rounds = max(1, video_max_rounds)
                
                # Get the video's label
                label = self.original_labels[video_idx]
                
                # Add mapping entries for augmented samples with the step size
                aug_rounds_added = 0
                for aug_round in range(1, video_max_rounds + 1, self.aug_step_size):
                    self.aug_video_map.append((video_idx, aug_round))
                    self.augmented_labels.append(label)
                    total_augmented_samples += 1
                    aug_rounds_added += 1
                
                if aug_rounds_added > 0:
                    videos_with_augmentation += 1
                
            except Exception as e:
                self.logger.error(f"Error setting up augmentation for {video_path}: {str(e)}")
        
        # Calculate class counts after augmentation
        class_counts_after = class_counts_before.copy()
        for label in self.augmented_labels:
            class_counts_after[label] = class_counts_after.get(label, 0) + 1
        
        # Log detailed augmentation info
        self.logger.info(f"Augmentation setup complete:")
        self.logger.info(f"  Videos processed: {videos_processed}")
        self.logger.info(f"  Videos with augmentation: {videos_with_augmentation}")
        self.logger.info(f"  Total augmented samples: {total_augmented_samples}")
        self.logger.info(f"  Dataset size after augmentation: {self.original_length + total_augmented_samples}")
        
        # Create class name mapping for nicer output
        class_names = {0: "non-referral", 1: "referral"} if len(set(self.original_labels)) == 2 else {}
        
        # Log detailed class distribution info
        original_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_before.items())])
        self.logger.info(f"Original class distribution: {original_dist_str}")
        
        augmented_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_after.items())])
        self.logger.info(f"Augmented class distribution: {augmented_dist_str}")
        
        # Calculate added samples per class
        added_counts = {k: class_counts_after.get(k, 0) - class_counts_before.get(k, 0) for k in set(class_counts_before) | set(class_counts_after)}
        added_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: +{v}" for k, v in sorted(added_counts.items())])
        self.logger.info(f"Added samples per class: {added_dist_str}")
    
    def calculate_max_aug_rounds(self, total_frames):
        """
        Calculate the maximum number of augmentation rounds based on video length and sampling method.
        
        Args:
            total_frames: Total number of frames in the video
            
        Returns:
            max_rounds: Maximum number of augmentation rounds
        """
        if not hasattr(self, 'num_frames'):
            # Default to 32 frames if not specified
            num_frames = 32
        else:
            num_frames = self.num_frames
            
        # Edge case: if num_frames is too large relative to total_frames,
        # the chunks will be too small for meaningful augmentation
        if num_frames > total_frames / 2:
            # When requesting too many frames, limit augmentation rounds
            return max(1, min(5, total_frames // (2 * num_frames) + 1))
        
        # For uniform sampling:
        if num_frames <= 1:
            return 1  # No augmentation possible with only 1 frame
        
        if self.sampling_method == 'uniform':
            # Calculate step using original formula for uniform sampling
            step = (total_frames - 1) / (num_frames - 1)
            
            # Find the minimum space between consecutive frames - this determines max rounds
            min_chunk_size = max(1, int(step) - 1)  # At least 1 frame between borders
            
            # Each augmentation round uses one frame from each chunk
            max_rounds = max(1, min_chunk_size)
            
        elif self.sampling_method == 'random':
            # For random sampling, we can generate multiple random samples
            # Limit to a reasonable number based on video length
            max_rounds = min(10, total_frames // num_frames)
            
        elif self.sampling_method == 'random_window':
            # For random window, we can select different frames from each window
            window_size = total_frames / num_frames
            if window_size < 1:
                max_rounds = 1  # Not enough frames for meaningful augmentation
            else:
                max_rounds = min(10, int(window_size))
        else:
            # Default to a reasonable value
            max_rounds = 5
            
        return max(1, max_rounds)
    
    def get_sampled_frames(self, video_path, total_frames, num_frames, sampling_method=None, aug_round=None):
        """
        Get sampled frame indices and track them for CSV export, with augmentation support.
        
        Args:
            video_path: Path to the video file
            total_frames: Total number of frames in the video
            num_frames: Number of frames to sample
            sampling_method: Override the default sampling method if specified
            aug_round: Augmentation round (None for original sampling)
            
        Returns:
            list: Selected frame indices
        """
        # Use specified sampling method or default
        method = sampling_method or self.sampling_method
        
        # Generate a cache key that includes augmentation round
        cache_key = f"{video_path}_{method}_{aug_round}"
        
        # Check if we already have cached indices for this video and aug round
        if cache_key in self.cached_indices:
            return self.cached_indices[cache_key]
        
        # Generate a deterministic seed based on the video path and augmentation round
        video_seed = 42  # Base seed for reproducibility
        if aug_round is not None:
            # Add augmentation round to seed for reproducible variation
            video_seed = video_seed + aug_round * 1000
        
        # Set random seed for this specific sampling
        random.seed(video_seed)
        np.random.seed(video_seed)
        torch.manual_seed(video_seed)
        
        # Get sampling indices based on method
        if method == 'uniform':
            indices = self._get_uniform_sampling_indices(total_frames, num_frames, aug_round)
        elif method == 'random':
            indices = self._get_random_sampling_indices(total_frames, num_frames, aug_round)
        elif method == 'random_window':
            indices = self._get_random_window_sampling_indices(total_frames, num_frames, aug_round)
        else:
            # Default to uniform sampling
            indices = self._get_uniform_sampling_indices(total_frames, num_frames, aug_round)
        
        # Reset the random seeds to the global seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Store the indices in cache
        self.cached_indices[cache_key] = indices
        
        return indices
    
    def _get_uniform_sampling_indices(self, total_frames, num_frames, aug_round=None):
        """
        Implement uniform sampling with augmentation support.
        
        Args:
            total_frames: Total frames in the video
            num_frames: Number of frames to sample
            aug_round: Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # Calculate border positions first (these are the same with or without augmentation)
        if num_frames == 1:
            border_indices = [total_frames // 2]  # Middle frame for single frame
        else:
            step = (total_frames - 1) / (num_frames - 1)
            border_indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
        
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
                    frame_idx = random.choice(available_frames)
                    round_frames.append(frame_idx)
                else:
                    # If no frames available, use the left border again
                    round_frames.append(chunk_start)
        
        # Always include the last border for completeness
        if border_indices:
            round_frames.append(border_indices[-1])
            
        return sorted(round_frames)
    
    def _get_random_sampling_indices(self, total_frames, num_frames, aug_round=None):
        """
        Implement random sampling with augmentation support.
        
        Args:
            total_frames: Total frames in the video
            num_frames: Number of frames to sample
            aug_round: Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # For both original and augmentation rounds, random sampling is similar,
        # but we use a different seed for each augmentation round which is handled by the caller
        
        if total_frames >= num_frames:
            # Random sampling without replacement when we have enough frames
            indices = sorted(random.sample(range(total_frames), num_frames))
        else:
            # Random sampling with replacement when we don't have enough frames
            indices = sorted(random.choices(range(total_frames), k=num_frames))
            
        return indices
    
    def _get_random_window_sampling_indices(self, total_frames, num_frames, aug_round=None):
        """
        Implement random window sampling with augmentation support.
        
        Args:
            total_frames: Total frames in the video
            num_frames: Number of frames to sample
            aug_round: Augmentation round (None for original sampling)
            
        Returns:
            list: Frame indices
        """
        # Calculate window size
        window_size = total_frames / num_frames
        indices = []
        
        # For each window, select one random frame
        for i in range(num_frames):
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
    
    def get_augmented_item(self, index):
        """
        Get an item from the dataset with augmentation support.
        
        This method should be called from the dataset's __getitem__ method
        and handles both original and augmented samples.
        
        Args:
            index: Dataset index
            
        Returns:
            tuple: (video_path, label, aug_round)
        """
        if not self.augment or index < self.original_length:
            # Original sample (no augmentation)
            video_path = self.original_video_paths[index]
            label = self.original_labels[index]
            aug_round = None
        else:
            # Augmented sample
            aug_idx = index - self.original_length
            if aug_idx < len(self.aug_video_map):
                video_idx, aug_round = self.aug_video_map[aug_idx]
                video_path = self.original_video_paths[video_idx]
                label = self.original_labels[video_idx]
            else:
                # Handle index out of range
                self.logger.warning(f"Augmentation index {aug_idx} out of range ({len(self.aug_video_map)})")
                video_path = self.original_video_paths[0]
                label = self.original_labels[0]
                aug_round = None
            
        return video_path, label, aug_round
    
    def save_sampling_indices(self):
        """
        Save the cached sampled indices to a CSV file.
        This ensures reproducibility and allows verification of sampling.
        """
        if not self.csv_save_dir:
            self.logger.warning("No CSV save directory provided, cannot save sampled indices")
            return
            
        # Create CSV filename based on mode and sampling method
        aug_info = f"_augstep{self.aug_step_size}" if self.augment else ""
        csv_path = os.path.join(self.csv_save_dir, f"sampling_indices_{self.dataset_name}_{self.sampling_method}_{self.split}{aug_info}.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['video_filename', 'total_frames', 'sampled_indices', 'aug_round', 'label'])
                
                # Write data for original samples
                if hasattr(self, 'original_video_paths'):
                    for i, video_path in enumerate(self.original_video_paths):
                        # Get the video's frame count
                        import cv2
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            self.logger.warning(f"Could not open video for sampling indices save: {video_path}")
                            continue
                            
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        # Make sure we have num_frames attribute
                        num_frames = self.num_frames if hasattr(self, 'num_frames') else 32
                        
                        # Get the original sampling indices
                        try:
                            original_indices = self.get_sampled_frames(
                                video_path, 
                                total_frames, 
                                num_frames,
                                self.sampling_method
                            )
                            
                            indices_str = ','.join(map(str, original_indices))
                            video_filename = os.path.basename(video_path)
                            label = self.original_labels[i] if hasattr(self, 'original_labels') else -1
                            
                            writer.writerow([video_filename, total_frames, indices_str, 'original', label])
                            
                            # Write augmentation rounds if augmentation is enabled
                            if self.augment:
                                # Calculate max rounds for this video
                                max_rounds = self.calculate_max_aug_rounds(total_frames)
                                if self.max_aug_rounds is not None:
                                    max_rounds = min(max_rounds, self.max_aug_rounds)
                                
                                # Write entries for each augmentation round using the step size
                                for aug_round in range(1, max_rounds + 1, self.aug_step_size):
                                    aug_indices = self.get_sampled_frames(
                                        video_path, 
                                        total_frames, 
                                        num_frames,
                                        self.sampling_method, 
                                        aug_round
                                    )
                                    
                                    indices_str = ','.join(map(str, aug_indices))
                                    writer.writerow([video_filename, total_frames, indices_str, f'aug_{aug_round}', label])
                        except Exception as e:
                            self.logger.error(f"Error processing video {video_path} for sampling indices: {str(e)}")
                
                self.logger.info(f"Saved sampling indices to {csv_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving sampling indices to CSV: {str(e)}")