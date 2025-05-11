# utils/augmentation_helper.py
import os
import random
import numpy as np
import torch
import cv2
import csv
from pathlib import Path

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def calculate_max_aug_rounds(total_frames, num_frames, sampling_method='uniform'):
    """
    Calculate the maximum number of augmentation rounds based on video length and sampling method.
    
    Args:
        total_frames: Total number of frames in the video
        num_frames: Number of frames to sample
        sampling_method: Sampling method ('uniform', 'random', or 'random_window')
        
    Returns:
        max_rounds: Maximum number of augmentation rounds
    """
    # Edge case: if num_frames is too large relative to total_frames,
    # the chunks will be too small for meaningful augmentation
    if num_frames > total_frames / 2:
        # When requesting too many frames, limit augmentation rounds
        return max(1, min(5, total_frames // (2 * num_frames) + 1))
    
    # For uniform sampling:
    if num_frames <= 1:
        return 1  # No augmentation possible with only 1 frame
    
    if sampling_method == 'uniform':
        # Calculate step using original formula for uniform sampling
        step = (total_frames - 1) / (num_frames - 1)
        
        # Find the minimum space between consecutive frames - this determines max rounds
        min_chunk_size = max(1, int(step) - 1)  # At least 1 frame between borders
        
        # Each augmentation round uses one frame from each chunk
        max_rounds = max(1, min_chunk_size)
        
    elif sampling_method == 'random':
        # For random sampling, we can generate multiple random samples
        # Limit to a reasonable number based on video length
        max_rounds = min(10, total_frames // num_frames)
        
    elif sampling_method == 'random_window':
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

def get_uniform_sampling_indices(total_frames, num_frames, aug_round=None, seed=42):
    """
    Get frame indices using uniform sampling with augmentation support.
    
    Args:
        total_frames: Total frames in the video
        num_frames: Number of frames to sample
        aug_round: Augmentation round (None for original sampling)
        seed: Random seed
        
    Returns:
        list: Selected frame indices
    """
    # Set seed for reproducibility
    random.seed(seed)
    
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

def get_random_sampling_indices(total_frames, num_frames, aug_round=None, seed=42):
    """
    Get frame indices using random sampling with augmentation support.
    
    Args:
        total_frames: Total frames in the video
        num_frames: Number of frames to sample
        aug_round: Augmentation round (None for original sampling)
        seed: Random seed
        
    Returns:
        list: Selected frame indices
    """
    # Set seed for reproducibility, modified by aug_round
    if aug_round is not None:
        seed = seed + aug_round * 1000
    random.seed(seed)
    
    if total_frames >= num_frames:
        # Random sampling without replacement when we have enough frames
        indices = sorted(random.sample(range(total_frames), num_frames))
    else:
        # Random sampling with replacement when we don't have enough frames
        indices = sorted(random.choices(range(total_frames), k=num_frames))
        
    return indices

def get_random_window_sampling_indices(total_frames, num_frames, aug_round=None, seed=42):
    """
    Get frame indices using random window sampling with augmentation support.
    
    Args:
        total_frames: Total frames in the video
        num_frames: Number of frames to sample
        aug_round: Augmentation round (None for original sampling)
        seed: Random seed
        
    Returns:
        list: Selected frame indices
    """
    # Set seed for reproducibility, modified by aug_round
    if aug_round is not None:
        seed = seed + aug_round * 1000
    random.seed(seed)
    
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

def get_sampling_indices(total_frames, num_frames, sampling_method, aug_round=None, seed=42):
    """
    Get frame indices based on sampling method with augmentation support.
    
    Args:
        total_frames: Total frames in the video
        num_frames: Number of frames to sample
        sampling_method: 'uniform', 'random', or 'random_window'
        aug_round: Augmentation round (None for original sampling)
        seed: Random seed
        
    Returns:
        list: Selected frame indices
    """
    # Set video-specific seed based on aug_round
    video_seed = seed
    if aug_round is not None:
        video_seed = seed + aug_round * 1000
    
    # Get sampling indices based on method
    if sampling_method == 'uniform':
        indices = get_uniform_sampling_indices(total_frames, num_frames, aug_round, video_seed)
    elif sampling_method == 'random':
        indices = get_random_sampling_indices(total_frames, num_frames, aug_round, video_seed)
    elif sampling_method == 'random_window':
        indices = get_random_window_sampling_indices(total_frames, num_frames, aug_round, video_seed)
    else:
        # Default to uniform sampling
        indices = get_uniform_sampling_indices(total_frames, num_frames, aug_round, video_seed)
    
    return indices

def setup_augmentation_for_dataset(dataset, logger, aug_step_size=1, max_aug_rounds=None):
    """
    Set up the augmentation for a dataset instance.
    
    Args:
        dataset: Dataset instance to augment
        logger: Logger to use
        aug_step_size: Step size for augmentation rounds
        max_aug_rounds: Maximum number of augmentation rounds
        
    Returns:
        dict: Augmentation statistics
    """
    logger.info(f"Setting up data augmentation with {dataset.sampling_method} sampling method "
               f"and step size {aug_step_size}")
    
    # Make sure original properties are set
    if not hasattr(dataset, 'original_video_paths'):
        dataset.original_video_paths = dataset._path_to_videos.copy()
    if not hasattr(dataset, 'original_labels'):
        dataset.original_labels = dataset._labels.copy()
    if not hasattr(dataset, 'original_length'):
        dataset.original_length = len(dataset.original_video_paths)
    
    # Create/clear augmentation maps
    dataset.aug_video_map = []
    dataset.augmented_labels = []
    dataset.cached_indices = {}
    
    # Process each video to calculate augmentation rounds and create mappings
    total_augmented_samples = 0
    videos_processed = 0
    videos_with_augmentation = 0
    
    # Count samples for each class before augmentation
    class_counts_before = {}
    for label in dataset.original_labels:
        class_counts_before[label] = class_counts_before.get(label, 0) + 1
    
    for video_idx, video_path in enumerate(dataset.original_video_paths):
        try:
            videos_processed += 1
            
            # Get video frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video for augmentation: {video_path}")
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Calculate max augmentation rounds for this video
            video_max_rounds = calculate_max_aug_rounds(
                total_frames, 
                dataset.cfg.DATA.NUM_FRAMES, 
                dataset.sampling_method
            )
            
            # Apply max rounds limit if specified
            if max_aug_rounds is not None:
                video_max_rounds = min(video_max_rounds, max_aug_rounds)
            
            # Make sure we have at least one round
            video_max_rounds = max(1, video_max_rounds)
            
            # Get the video's label
            label = dataset.original_labels[video_idx]
            
            # Add mapping entries for augmented samples with the step size
            aug_rounds_added = 0
            for aug_round in range(1, video_max_rounds + 1, aug_step_size):
                dataset.aug_video_map.append((video_idx, aug_round))
                dataset.augmented_labels.append(label)
                total_augmented_samples += 1
                aug_rounds_added += 1
            
            if aug_rounds_added > 0:
                videos_with_augmentation += 1
            
        except Exception as e:
            logger.error(f"Error setting up augmentation for {video_path}: {str(e)}")
    
    # Calculate class counts after augmentation
    class_counts_after = class_counts_before.copy()
    for label in dataset.augmented_labels:
        class_counts_after[label] = class_counts_after.get(label, 0) + 1
    
    # Log detailed augmentation info
    logger.info(f"Augmentation setup complete:")
    logger.info(f"  Videos processed: {videos_processed}")
    logger.info(f"  Videos with augmentation: {videos_with_augmentation}")
    logger.info(f"  Total augmented samples: {total_augmented_samples}")
    logger.info(f"  Dataset size after augmentation: {dataset.original_length + total_augmented_samples}")
    
    # Create class name mapping for nicer output
    class_names = {0: "non-referral", 1: "referral"} if len(set(dataset.original_labels)) == 2 else {}
    
    # Log detailed class distribution info
    original_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_before.items())])
    logger.info(f"Original class distribution: {original_dist_str}")
    
    augmented_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_after.items())])
    logger.info(f"Augmented class distribution: {augmented_dist_str}")
    
    # Calculate added samples per class
    added_counts = {k: class_counts_after.get(k, 0) - class_counts_before.get(k, 0) for k in set(class_counts_before) | set(class_counts_after)}
    added_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: +{v}" for k, v in sorted(added_counts.items())])
    logger.info(f"Added samples per class: {added_dist_str}")
    
    return {
        'original_samples': dataset.original_length,
        'augmented_samples': total_augmented_samples,
        'total_samples': dataset.original_length + total_augmented_samples,
        'videos_with_augmentation': videos_with_augmentation
    }

def get_augmented_item(dataset, index):
    """
    Get an item from the dataset with augmentation support.
    
    Args:
        dataset: Dataset instance
        index: Item index
        
    Returns:
        tuple: (original_index, aug_round) - original_index for __getitem__, aug_round for frame sampling
    """
    if not hasattr(dataset, 'augment') or not dataset.augment or index < dataset.original_length:
        # Original sample (no augmentation)
        return index, None
    else:
        # Augmented sample
        aug_idx = index - dataset.original_length
        if aug_idx < len(dataset.aug_video_map):
            video_idx, aug_round = dataset.aug_video_map[aug_idx]
            return video_idx, aug_round
        else:
            # Handle index out of range
            dataset.logger.warning(f"Augmentation index {aug_idx} out of range ({len(dataset.aug_video_map)})")
            return 0, None

def save_sampling_indices(dataset, csv_dir):
    """
    Save the sampled indices to a CSV file for reproducibility.
    
    Args:
        dataset: Dataset instance
        csv_dir: Directory to save CSV file
    """
    if not csv_dir:
        if hasattr(dataset, 'logger'):
            dataset.logger.warning("No CSV save directory provided, cannot save sampled indices")
        else:
            print("No CSV save directory provided, cannot save sampled indices")
        return
        
    # Create CSV filename with augmentation info
    aug_info = f"_augstep{dataset.aug_step_size}" if hasattr(dataset, 'augment') and dataset.augment else ""
    dataset_name = getattr(dataset, 'dataset_name', 'dataset')
    sampling_method = getattr(dataset, 'sampling_method', 'unknown')
    split = getattr(dataset, 'split', 'unknown')
    
    csv_path = os.path.join(csv_dir, f"sampling_indices_{dataset_name}_{sampling_method}_{split}{aug_info}.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['video_filename', 'total_frames', 'sampled_indices', 'aug_round', 'label'])
            
            # Write data for original samples
            if hasattr(dataset, 'original_video_paths'):
                for i, video_path in enumerate(dataset.original_video_paths):
                    try:
                        # Get the video's frame count
                        import cv2
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            if hasattr(dataset, 'logger'):
                                dataset.logger.warning(f"Could not open video for sampling indices save: {video_path}")
                            else:
                                print(f"Could not open video for sampling indices save: {video_path}")
                            continue
                            
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        # Make sure we have num_frames attribute
                        num_frames = getattr(dataset, 'num_frames', 32)
                        if hasattr(dataset, 'cfg') and hasattr(dataset.cfg, 'DATA') and hasattr(dataset.cfg.DATA, 'NUM_FRAMES'):
                            num_frames = dataset.cfg.DATA.NUM_FRAMES
                        
                        # Get the original sampling indices
                        original_indices = get_sampling_indices(
                            total_frames, 
                            num_frames,
                            sampling_method
                        )
                        
                        indices_str = ','.join(map(str, original_indices))
                        video_filename = os.path.basename(video_path)
                        label = dataset.original_labels[i] if hasattr(dataset, 'original_labels') else -1
                        
                        writer.writerow([video_filename, total_frames, indices_str, 'original', label])
                        
                        # Write augmentation rounds if augmentation is enabled
                        if hasattr(dataset, 'augment') and dataset.augment:
                            # Calculate max rounds for this video
                            max_rounds = calculate_max_aug_rounds(
                                total_frames, 
                                num_frames,
                                sampling_method
                            )
                            
                            if hasattr(dataset, 'max_aug_rounds') and dataset.max_aug_rounds is not None:
                                max_rounds = min(max_rounds, dataset.max_aug_rounds)
                            
                            # Get step size
                            step_size = getattr(dataset, 'aug_step_size', 1)
                            
                            # Write entries for each augmentation round
                            for aug_round in range(1, max_rounds + 1, step_size):
                                aug_indices = get_sampling_indices(
                                    total_frames, 
                                    num_frames,
                                    sampling_method, 
                                    aug_round
                                )
                                
                                indices_str = ','.join(map(str, aug_indices))
                                writer.writerow([video_filename, total_frames, indices_str, f'aug_{aug_round}', label])
                    except Exception as e:
                        if hasattr(dataset, 'logger'):
                            dataset.logger.error(f"Error processing video {video_path} for sampling indices: {str(e)}")
                        else:
                            print(f"Error processing video {video_path} for sampling indices: {str(e)}")
        
        if hasattr(dataset, 'logger'):
            dataset.logger.info(f"Saved sampling indices to {csv_path}")
        else:
            print(f"Saved sampling indices to {csv_path}")
            
    except Exception as e:
        if hasattr(dataset, 'logger'):
            dataset.logger.error(f"Error saving sampling indices to CSV: {str(e)}")
        else:
            print(f"Error saving sampling indices to CSV: {str(e)}")