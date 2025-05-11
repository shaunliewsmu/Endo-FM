"""
Modified UCF101 class with integrated data augmentation support.
Replace your existing UCF101 class with this implementation.
"""

import os
import random
import warnings
import numpy as np
import logging
import torch
import torch.utils.data
import traceback

from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling
from datasets.decoder import decode
from datasets.video_container import get_video_container


class UCF101(torch.utils.data.Dataset):
    """
    UCF101 video loader with integrated data augmentation support.
    This class supports:
    1. Regular video loading for train/val/test
    2. Data augmentation through multiple sampling rounds
    3. Consistent frame sampling based on configurable methods
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the UCF101 video loader with a given csv file.
        
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], f"Split '{mode}' not supported for UCF101"
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._split_idx = mode
        
        # For training mode, one single clip is sampled from every video.
        # For validation or testing, NUM_ENSEMBLE_VIEWS clips are sampled.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["val", "test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        # Initialize augmentation properties
        self.augment = False
        self.aug_video_map = []  # Will store tuples of (video_idx, aug_round)
        self.original_length = 0
        self.aug_step_size = 1
        self.sampling_method = getattr(cfg.DATA, f"{mode.upper()}_SAMPLING_METHOD", "uniform")
        self.max_aug_rounds = None
        self.dataset_name = "ucf101"
        self.csv_save_dir = None
        
        print(f"Constructing {mode}...")
        self._construct_loader()
        
        # Save original video paths and length
        self.original_video_paths = self._path_to_videos.copy()
        self.original_length = len(self._path_to_videos)

    def init_sampler(self, csv_save_dir, logger, dataset_name, split, sampling_method, 
                      augment=False, max_aug_rounds=None, aug_step_size=1):
        """Initialize the frame sampler with specified parameters"""
        self.csv_save_dir = csv_save_dir
        self.dataset_name = dataset_name
        self.split = split
        self.sampling_method = sampling_method
        self.augment = augment
        self.max_aug_rounds = max_aug_rounds
        self.aug_step_size = aug_step_size
        
        # Only setup augmentation for training
        if augment and split == "train":
            logger.info(f"Setting up data augmentation with {sampling_method} sampling method and step size {aug_step_size}")
            self._setup_augmentation(logger)

    def save_sampling_indices(self):
        """Save sampling indices to CSV file for reproducibility"""
        if not self.csv_save_dir:
            return
            
        import os
        import pandas as pd
        
        # Create sampling indices CSV file
        csv_path = os.path.join(
            self.csv_save_dir, 
            f"sampling_indices_{self.dataset_name}_{self.sampling_method}_{self.split}"
            f"{'_augstep' + str(self.aug_step_size) if self.augment else ''}.csv"
        )
        
        # Build dataframe with video paths and sampling indices
        data = []
        for i, path in enumerate(self.original_video_paths):
            # Get video information
            try:
                video_container = get_video_container(path, False, self.cfg.DATA.DECODING_BACKEND)
                total_frames = video_container.streams.video[0].frames
                
                # Get original sampling indices
                indices = self._get_sampling_indices(total_frames, i, None)
                
                # Add to dataframe
                data.append({
                    'video_path': path,
                    'total_frames': total_frames,
                    'sampling_method': self.sampling_method,
                    'indices': str(indices)
                })
                
                # If augmentation is enabled, add augmented indices
                if self.augment and self.mode == "train":
                    # Determine number of augmentation rounds
                    max_rounds = self._calculate_max_rounds(total_frames)
                    
                    # Add each augmentation round
                    for round_idx in range(1, max_rounds+1, self.aug_step_size):
                        aug_indices = self._get_sampling_indices(total_frames, i, round_idx)
                        data.append({
                            'video_path': path,
                            'total_frames': total_frames,
                            'sampling_method': self.sampling_method,
                            'aug_round': round_idx,
                            'indices': str(aug_indices)
                        })
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
                
        # Create and save dataframe
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Saved sampling indices to {csv_path}")

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, f"{self.mode}.txt"
        )
        assert os.path.exists(path_to_file), f"{path_to_file} dir not found"

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) > 0), f"Failed to load UCF101 split {self._split_idx} from {path_to_file}"
        print(f"Constructing dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

    def _setup_augmentation(self, logger):
        """
        Set up data augmentation for the dataset.
        """
        # Initialize augmentation tracking
        self.aug_video_map = []
        class_distributions = {
            'original': {},
            'augmented': {},
        }
        
        # Track original class distribution
        for i, label in enumerate(self._labels):
            if label not in class_distributions['original']:
                class_distributions['original'][label] = 0
            class_distributions['original'][label] += 1
        
        # Process each video for augmentation
        videos_processed = 0
        videos_with_augmentation = 0
        
        # Process videos up to the original length only
        for i in range(min(self.original_length, len(self._path_to_videos))):
            video_path = self._path_to_videos[i]
            video_label = self._labels[i]
            
            try:
                # Open video to get frame count
                video_container = get_video_container(video_path, False, self.cfg.DATA.DECODING_BACKEND)
                total_frames = video_container.streams.video[0].frames
                
                # Calculate maximum augmentation rounds
                max_rounds = self._calculate_max_rounds(total_frames)
                
                # Add augmentation entries for this video
                aug_rounds_added = 0
                for round_idx in range(1, max_rounds+1, self.aug_step_size):
                    # Create a proper 2-element tuple
                    self.aug_video_map.append((i, round_idx))
                    aug_rounds_added += 1
                    
                    # Track augmented class distribution
                    if video_label not in class_distributions['augmented']:
                        class_distributions['augmented'][video_label] = 0
                    class_distributions['augmented'][video_label] += 1
                
                if aug_rounds_added > 0:
                    videos_with_augmentation += 1
                    
                videos_processed += 1
                
            except Exception as e:
                logger.warning(f"Failed to process video for augmentation: {video_path}, error: {str(e)}")
        
        # Log augmentation statistics
        logger.info(f"Augmentation setup complete:")
        logger.info(f"  Videos processed: {videos_processed}")
        logger.info(f"  Videos with augmentation: {videos_with_augmentation}")
        logger.info(f"  Total augmented samples: {len(self.aug_video_map)}")
        logger.info(f"  Dataset size after augmentation: {self.original_length + len(self.aug_video_map)}")
        
        # Verify a few entries of aug_video_map for debugging
        if len(self.aug_video_map) > 0:
            logger.info(f"Sample aug_video_map entries:")
            for i in range(min(5, len(self.aug_video_map))):
                logger.info(f"  Entry {i}: {self.aug_video_map[i]}")
        
        # Log class distribution
        class_labels = {0: 'non-referral', 1: 'referral'}  # Adjust based on your dataset
        
        # Original distribution
        orig_dist_str = ", ".join([f"{class_labels.get(k, k)}: {v}" for k, v in class_distributions['original'].items()])
        logger.info(f"Original class distribution: {orig_dist_str}")
        
        # Augmented distribution
        aug_dist = {}
        for k in set(list(class_distributions['original'].keys()) + list(class_distributions['augmented'].keys())):
            aug_dist[k] = class_distributions['original'].get(k, 0) + class_distributions['augmented'].get(k, 0)
        
        aug_dist_str = ", ".join([f"{class_labels.get(k, k)}: {v}" for k, v in aug_dist.items()])
        logger.info(f"Augmented class distribution: {aug_dist_str}")
        
        # Added samples
        added_samples = {}
        for k in class_distributions['augmented']:
            added_samples[k] = class_distributions['augmented'][k]
        
        added_str = ", ".join([f"{class_labels.get(k, k)}: +{v}" for k, v in added_samples.items()])
        logger.info(f"Added samples per class: {added_str}")

    def _calculate_max_rounds(self, total_frames):
        """
        Calculate the maximum number of augmentation rounds based on video length.
        """
        # If max rounds is explicitly set, use that
        if self.max_aug_rounds is not None:
            return self.max_aug_rounds
        
        # Otherwise calculate based on video length and frame count
        frames_needed = self.cfg.DATA.NUM_FRAMES
        
        # For very short videos, use a fixed small number
        if total_frames < frames_needed * 2:
            return 3
            
        # For longer videos, scale with length
        max_rounds = min(30, total_frames // 2)  # Cap at 30 rounds
        return max_rounds

    def _get_sampling_indices(self, total_frames, video_idx=None, aug_round=None):
        """
        Get sampling indices for a video based on sampling method and augmentation round.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        method = self.sampling_method
        
        # Set random seed for reproducibility if provided
        if hasattr(self.cfg.DATA, 'SEED') and self.cfg.DATA.SEED is not None:
            seed = self.cfg.DATA.SEED
            if video_idx is not None and aug_round is not None:
                # Different seed for each video and augmentation round
                seed = seed + video_idx * 1000 + aug_round * 100
                
            # Set seed
            np.random.seed(seed)
            random.seed(seed)
        
        # Calculate indices based on sampling method
        indices = []
        
        if method == 'uniform':
            # Uniform sampling (evenly spaced frames)
            if total_frames <= num_frames:
                # If video is shorter than required frames, repeat frames
                indices = np.arange(total_frames)
                # Repeat the last frame to reach desired count
                indices = np.pad(indices, (0, num_frames - total_frames), 'edge')
            else:
                # Calculate stride for uniform sampling
                stride = total_frames / float(num_frames)
                # Apply augmentation offset if needed
                offset = 0
                if aug_round is not None:
                    offset = aug_round % stride
                # Generate indices with offset
                indices = np.array([int(offset + stride * i) for i in range(num_frames)])
                # Ensure we don't exceed video length
                indices = np.clip(indices, 0, total_frames - 1)
                
        elif method == 'random':
            # Random sampling (random selection of frames)
            if total_frames <= num_frames:
                # If video is shorter than required frames, repeat frames
                indices = np.arange(total_frames)
                # Repeat the last frame to reach desired count
                indices = np.pad(indices, (0, num_frames - total_frames), 'edge')
            else:
                # Set specific seed for this video and augmentation round
                if video_idx is not None and aug_round is not None:
                    state = np.random.get_state()
                    np.random.seed(self.cfg.DATA.SEED + video_idx * 1000 + aug_round)
                
                # Random selection without replacement
                indices = np.sort(np.random.choice(total_frames, num_frames, replace=False))
                
                # Restore random state if needed
                if video_idx is not None and aug_round is not None:
                    np.random.set_state(state)
                    
        elif method == 'random_window':
            # Random window sampling (consecutive frames from random start)
            if total_frames <= num_frames:
                # If video is shorter than required frames, repeat frames
                indices = np.arange(total_frames)
                # Repeat the last frame to reach desired count
                indices = np.pad(indices, (0, num_frames - total_frames), 'edge')
            else:
                # Determine window start
                max_start = total_frames - num_frames
                
                # Set specific seed for this video and augmentation round
                if video_idx is not None and aug_round is not None:
                    state = np.random.get_state()
                    np.random.seed(self.cfg.DATA.SEED + video_idx * 1000 + aug_round)
                
                # Select random start point with offset if augmenting
                if aug_round is not None:
                    # Divide max_start into regions for better augmentation diversity
                    region_size = max(1, max_start // 30)
                    start_idx = min(max_start, (aug_round % 30) * region_size)
                else:
                    # Random start point for original sampling
                    start_idx = np.random.randint(0, max_start + 1)
                
                # Restore random state if needed
                if video_idx is not None and aug_round is not None:
                    np.random.set_state(state)
                
                # Generate consecutive indices
                indices = np.arange(start_idx, start_idx + num_frames)
                indices = np.clip(indices, 0, total_frames - 1)  # Safety clip
        else:
            raise ValueError(f"Unsupported sampling method: {method}")
            
        return indices

    def __getitem__(self, index):
        """
        Get a video clip with option for data augmentation.
        
        Args:
            index (int): the video index provided by the pytorch sampler.
            
        Returns:
            frames (tensor): the frames sampled from the video.
            label (int): the label of the current video.
            index (int): the index of the video.
            meta (dict): additional metadata.
        """
        logger = logging.getLogger('endo_fm')
        original_index = index
        
        try:
            # Handle augmented indices
            if self.augment and index >= self.original_length:
                # Get the augmentation index
                aug_idx = index - self.original_length
                if aug_idx >= len(self.aug_video_map):
                    aug_idx = aug_idx % len(self.aug_video_map)
                
                # Get original video index and augmentation round
                video_idx, aug_round = self.aug_video_map[aug_idx]
            else:
                # Original sample
                video_idx = index
                aug_round = None
                
            # Make sure video_idx is in range
            video_idx = video_idx % len(self._path_to_videos)
            
            # Set up parameters for decoding
            if self.mode in ["train"]:
                temporal_sample_index = -1
                spatial_sample_index = -1
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
                max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            elif self.mode in ["val", "test"]:
                # Get spatial_temporal_idx safely
                st_idx = min(video_idx, len(self._spatial_temporal_idx) - 1)
                spatial_temporal_idx = self._spatial_temporal_idx[st_idx]
                
                temporal_sample_index = (spatial_temporal_idx // self.cfg.TEST.NUM_SPATIAL_CROPS)
                spatial_sample_index = (
                    (spatial_temporal_idx % self.cfg.TEST.NUM_SPATIAL_CROPS)
                    if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
                )
                
                min_scale, max_scale, crop_size = (
                    [self.cfg.DATA.TEST_CROP_SIZE] * 3 
                    if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                    else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
                )
            else:
                raise NotImplementedError(f"Does not support {self.mode} mode")
            
            # Get sampling rate
            sampling_rate = get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            
            # Try to decode the video
            for i_try in range(self._num_retries):
                try:
                    # Get video path
                    path = self._path_to_videos[video_idx]
                    
                    # Load video container
                    video_container = get_video_container(
                        path,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                    
                    if video_container is None:
                        continue
                    
                    # Get a safe meta_idx for video_meta
                    meta_idx = min(video_idx, len(self._video_meta) - 1)
                    
                    # Get total frames for this video
                    total_frames = video_container.streams.video[0].frames
                    
                    # We'll use a modified approach for augmentation - instead of trying to 
                    # extract frames directly, we'll manipulate the temporal_sample_index
                    # to get different frames for each augmentation round
                    if aug_round is not None:
                        # For uniform sampling, we directly manipulate the clip start point
                        if self.sampling_method == 'uniform':
                            # Calculate clip start offset based on augmentation round
                            clip_size = self.cfg.DATA.NUM_FRAMES * sampling_rate
                            max_offset = max(1, total_frames - clip_size)
                            
                            # Scale offset by augmentation round (0-1 range)
                            offset_factor = (aug_round % 30) / 30.0
                            clip_start = int(offset_factor * max_offset)
                            
                            # Save to video_meta for the decoder to use
                            self._video_meta[meta_idx]['clip_start_frame'] = clip_start
                        
                        # For random sampling, we'll use a deterministic seed based on aug_round
                        elif self.sampling_method in ['random', 'random_window']:
                            # Set seed for deterministic randomness
                            seed = self.cfg.DATA.SEED + video_idx * 1000 + aug_round * 100
                            
                            # Save seed to video_meta for the decoder to use
                            self._video_meta[meta_idx]['sample_seed'] = seed
                    
                    # Decode frames
                    frames = decode(
                        container=video_container,
                        sampling_rate=sampling_rate,
                        num_frames=self.cfg.DATA.NUM_FRAMES,
                        clip_idx=temporal_sample_index,
                        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                        video_meta=self._video_meta[meta_idx],
                        target_fps=self.cfg.DATA.TARGET_FPS,
                        backend=self.cfg.DATA.DECODING_BACKEND,
                        max_spatial_scale=min_scale,
                    )
                    
                    if frames is None:
                        if i_try < self._num_retries - 1:
                            continue
                        else:
                            # Last retry failed, use fallback
                            logger.warning(f"All retries failed for video {path}")
                            # Create a random tensor as fallback
                            frames = np.random.randint(0, 256, (
                                self.cfg.DATA.NUM_FRAMES, 
                                crop_size, 
                                crop_size, 
                                3
                            )).astype(np.uint8)
                    
                    # Get label
                    label = self._labels[video_idx]
                    
                    # Process frames
                    frames = tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                    frames = frames.permute(3, 0, 1, 2)
                    
                    # Apply spatial sampling
                    frames = spatial_sampling(
                        frames,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                    
                    # Successful processing
                    return frames, label, original_index, {}
                    
                except Exception as e:
                    logger.error(f"Error on try {i_try}: {str(e)}")
                    if i_try == self._num_retries - 1:
                        logger.error(traceback.format_exc())
                    
                    # Try another video if running out of retries
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        video_idx = random.randint(0, self.original_length - 1)
                    
            # If we've exhausted all retries, return a random tensor
            logger.error(f"Failed to fetch video after {self._num_retries} retries. Returning random tensor.")
            random_tensor = torch.rand(3, self.cfg.DATA.NUM_FRAMES, crop_size, crop_size)
            
            # Use a valid label
            label = self._labels[video_idx % len(self._labels)]
            
            return random_tensor, label, original_index, {}
            
        except Exception as e:
            # Global exception handler
            logger.error(f"Unexpected error in __getitem__: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Emergency fallback - return random data
            random_tensor = torch.rand(3, self.cfg.DATA.NUM_FRAMES, 
                                       self.cfg.DATA.TRAIN_CROP_SIZE, 
                                       self.cfg.DATA.TRAIN_CROP_SIZE)
            
            # Get a valid label using a safe index
            safe_idx = index % len(self._labels)
            label = self._labels[safe_idx]
            
            return random_tensor, label, original_index, {}

    def __len__(self):
        """
        Returns the effective size of the dataset including augmented samples.
        """
        if self.augment and hasattr(self, 'aug_video_map'):
            return self.original_length + len(self.aug_video_map)
        return len(self._path_to_videos)