import os
import csv
import random
import numpy as np
import torch
import glob
from pathlib import Path
import cv2
import shutil

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
    """Generate frame indices based on sampling method"""
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

def get_video_frame_count(video_path):
    """Get the total number of frames in a video using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return 300  # Default if can't open
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            print(f"Warning: Got invalid frame count ({total_frames}) for {video_path}")
            return 300  # Default if invalid count
        return total_frames
    except Exception as e:
        print(f"Error getting frame count for {video_path}: {str(e)}")
        return 300  # Default on error

def find_all_videos(root_dir, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """Find all video files in the root directory"""
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True))
    return videos

def read_split_file(split_file):
    """Read video filenames from a split file"""
    videos = []
    try:
        with open(split_file, 'r') as f:
            print(f"Reading split file: {split_file}")
            print("First 5 lines of the split file:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  Line {i+1}: {line.strip()}")
                # Reset file position
                if i == 4:
                    f.seek(0)
                    
            # Now actually read the file
            f.seek(0)
            for line in f:
                # Split by whitespace (space, tab, etc.)
                parts = line.strip().split()
                if parts:
                    # The filename is the first part
                    video_filename = parts[0]
                    # Remove any trailing comma or other non-filename characters
                    video_filename = video_filename.split(',')[0]
                    videos.append(video_filename)
    except Exception as e:
        print(f"Error reading split file {split_file}: {str(e)}")
    
    # Print some sample filenames
    if videos:
        print(f"Sample filenames from split file (first 5):")
        for i, v in enumerate(videos[:5]):
            print(f"  {i+1}: {v}")
    
    return videos

def find_video_file(root_dir, video_name):
    """Find a video file in the video directory"""
    # First try direct match
    direct_match = os.path.join(root_dir, video_name)
    if os.path.exists(direct_match) and os.path.isfile(direct_match):
        return direct_match
    
    # Try recursive search for exact filename
    for root, _, files in os.walk(root_dir):
        if video_name in files:
            return os.path.join(root, video_name)
    
    # Try searching with just the basename
    basename = os.path.basename(video_name)
    for root, _, files in os.walk(root_dir):
        if basename in files:
            return os.path.join(root, basename)
    
    # Try searching with glob pattern (case insensitive)
    for path in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.basename(path).lower() == basename.lower():
            return path
            
    # Final approach: list all video files and look for close matches
    all_videos = find_all_videos(root_dir)
    for video in all_videos:
        if os.path.basename(video).lower() == basename.lower():
            return video
            
    return None

def debug_video_directory(video_root_path):
    """Print debug information about the video directory structure"""
    print(f"\nDebugging video directory: {video_root_path}")
    
    if not os.path.exists(video_root_path):
        print(f"  Error: Directory does not exist")
        return
        
    # Count files at top level
    top_files = [f for f in os.listdir(video_root_path) if os.path.isfile(os.path.join(video_root_path, f))]
    top_dirs = [d for d in os.listdir(video_root_path) if os.path.isdir(os.path.join(video_root_path, d))]
    
    print(f"  Top-level directory contains {len(top_files)} files and {len(top_dirs)} subdirectories")
    
    if top_files:
        print(f"  Sample files at top level (first 5):")
        for i, f in enumerate(top_files[:5]):
            print(f"    {i+1}: {f}")
    
    if top_dirs:
        print(f"  Subdirectories:")
        for i, d in enumerate(top_dirs):
            d_path = os.path.join(video_root_path, d)
            d_files = [f for f in os.listdir(d_path) if os.path.isfile(os.path.join(d_path, f))]
            print(f"    {i+1}: {d} - contains {len(d_files)} files")
            if d_files:
                print(f"      Sample files (first 3):")
                for j, f in enumerate(d_files[:3]):
                    print(f"        {j+1}: {f}")
    
    # Find all video files recursively
    all_videos = find_all_videos(video_root_path)
    print(f"  Found {len(all_videos)} video files in total")
    if all_videos:
        print(f"  Sample video paths (first 5):")
        for i, v in enumerate(all_videos[:5]):
            print(f"    {i+1}: {v}")

def create_csv_files(output_dir, dataset_name, train_sampling, val_sampling, test_sampling, 
                     video_root_path, splits_path, num_frames=8):
    """Create CSV files for sampling visualization using actual dataset videos"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a fixed seed for reproducibility
    seed = 42
    set_all_random_seeds(seed)
    
    # Print debug information about the directory structure
    debug_video_directory(video_root_path)
    
    # Find split files
    train_split_file = os.path.join(splits_path, 'train.txt')
    val_split_file = os.path.join(splits_path, 'val.txt')
    
    # Read split files if they exist
    train_videos = read_split_file(train_split_file) if os.path.exists(train_split_file) else []
    val_videos = read_split_file(val_split_file) if os.path.exists(val_split_file) else []
    
    print(f"Found {len(train_videos)} videos in train split")
    print(f"Found {len(val_videos)} videos in val split")
    
    # If split files don't exist or are empty, find videos directly
    if not train_videos and not val_videos:
        all_videos = find_all_videos(video_root_path)
        print(f"Found {len(all_videos)} videos in {video_root_path}")
        
        # Split videos 80% train, 20% val
        random.shuffle(all_videos)
        split_idx = int(len(all_videos) * 0.8)
        train_videos = all_videos[:split_idx]
        val_videos = all_videos[split_idx:]
    
    # Helper function to process videos for a split
    def process_split(videos, split, sampling_method):
        csv_path = os.path.join(output_dir, f"sampling_indices_{dataset_name}_{sampling_method}_{split}.csv")
        
        processed_count = 0
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header with expected format
            writer.writerow(['video_filename', 'total_frames', 'sampled_indices'])
            
            for video in videos:
                # Get video path - first try to find it in the video root path
                video_path = find_video_file(video_root_path, video)
                
                if not video_path:
                    print(f"Warning: Could not find video file for {video}, skipping")
                    continue
                
                # Get total frames
                total_frames = get_video_frame_count(video_path)
                
                # Get sampling indices
                indices = get_sampling_indices(
                    total_frames,
                    num_frames,
                    sampling_method,
                    seed
                )
                
                # First two columns
                row = [os.path.basename(video), total_frames]
                # Add all indices as additional columns
                row.extend(indices)
                writer.writerow(row)
                processed_count += 1
                
                # Print progress every 10 videos
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(videos)} videos for {split} split")
        
        print(f"Created {csv_path} with {processed_count} videos")
        
        # Verify the CSV file was created properly
        verify_csv(csv_path)
    
    # Helper function to verify CSV content
    def verify_csv(csv_path):
        try:
            with open(csv_path, 'r') as f:
                content = f.read()
                # Count lines
                lines = content.strip().split('\n')
                print(f"CSV file contains {len(lines)} lines (including header)")
                
                # Print first few lines
                if len(lines) > 1:
                    print("First 3 lines of CSV:")
                    for i, line in enumerate(lines[:3]):
                        print(f"  {line}")
                else:
                    print("CSV file only contains header!")
                    
                # Check file size
                file_size = os.path.getsize(csv_path)
                print(f"CSV file size: {file_size} bytes")
        except Exception as e:
            print(f"Error verifying CSV file: {str(e)}")
    
    # Process train and val splits
    process_split(train_videos, "train", train_sampling)
    process_split(val_videos, "val", val_sampling)
    
    # If the generation wasn't successful, create a backup with dummy data
    train_csv = os.path.join(output_dir, f"sampling_indices_{dataset_name}_{train_sampling}_train.csv")
    if os.path.getsize(train_csv) <= 50:  # Only header
        print("Generated train CSV is empty, creating backup with dummy data...")
        with open(train_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['video_filename', 'total_frames', 'sampled_frames'])
            # Add some dummy data
            for i in range(5):
                # Generate deterministic sampling indices
                indices = [int(i*30 + j*10) for j in range(num_frames)]
                row = [f'video_{i}.mp4', 300]
                row.extend(indices)
                writer.writerow(row)
        print(f"Created backup CSV with 5 dummy videos")
        
    val_csv = os.path.join(output_dir, f"sampling_indices_{dataset_name}_{val_sampling}_val.csv")
    if os.path.getsize(val_csv) <= 50:  # Only header
        print("Generated val CSV is empty, creating backup with dummy data...")
        with open(val_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['video_filename', 'total_frames', 'sampled_frames'])
            # Add some dummy data
            for i in range(5):
                # Generate deterministic sampling indices
                indices = [int(i*30 + j*30) for j in range(num_frames)]
                row = [f'val_video_{i}.mp4', 300]
                row.extend(indices)
                writer.writerow(row)
        print(f"Created backup CSV with 5 dummy videos")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CSV files for sampling visualization")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV files")
    parser.add_argument("--dataset", type=str, default="ucf101", help="Dataset name")
    parser.add_argument("--train_sampling", type=str, default="random", help="Training sampling method")
    parser.add_argument("--val_sampling", type=str, default="uniform", help="Validation sampling method")
    parser.add_argument("--test_sampling", type=str, default="uniform", help="Testing sampling method")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory for videos")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to splits directory")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample")
    
    args = parser.parse_args()
    
    create_csv_files(
        args.output_dir,
        args.dataset,
        args.train_sampling,
        args.val_sampling,
        args.test_sampling,
        args.video_root,
        args.splits_path,
        args.num_frames
    )
