import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import glob
import sys
from pathlib import Path

def find_video_file(root_dir, filename):
    """Search for a video file in the root directory"""
    # Try direct match
    direct_match = os.path.join(root_dir, filename)
    if os.path.exists(direct_match):
        return direct_match
    
    # Try recursive search
    for path in Path(root_dir).rglob(filename):
        return str(path)
    
    # Try without extension if not found
    name_without_ext = os.path.splitext(filename)[0]
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for path in Path(root_dir).rglob(f"{name_without_ext}{ext}"):
            return str(path)
    
    return None

def create_sampling_visualization(video_path, sampled_indices, sampling_method, seed, output_path):
    """Create a visualization of the sampled frames from a video"""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default to 30fps if not available
    
    # Create figure with 2 rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                  gridspec_kw={'height_ratios': [1, 3]})
    
    # Top row: sampling pattern visualization
    ax1.plot([0, total_frames], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, 0], [-0.2, 0.2], 'k-', linewidth=2)
    ax1.plot([total_frames, total_frames], [-0.2, 0.2], 'k-', linewidth=2)
    
    # Mark sampled frames
    for i, frame_idx in enumerate(sampled_indices):
        color = plt.cm.rainbow(i / len(sampled_indices))
        ax1.plot([frame_idx, frame_idx], [-0.3, 0.3], '-', color=color, linewidth=2)
        ax1.plot(frame_idx, 0, 'o', color=color, markersize=8)
        ax1.text(frame_idx, 0.1, f'{frame_idx}', 
              horizontalalignment='center', color=color, fontsize=9)
    
    # Customize plot
    ax1.set_title(f"Sampling Pattern: {sampling_method} - {len(sampled_indices)} frames from {total_frames} total (seed: {seed})")
    ax1.set_xlabel('Frame Index')
    ax1.set_xlim(-total_frames*0.05, total_frames*1.05)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    
    # Display time marks
    time_marks = 5
    for i in range(time_marks + 1):
        frame = int(i * total_frames / time_marks)
        time = frame / fps
        ax1.text(frame, -0.2, f'{time:.1f}s', 
              horizontalalignment='center', color='black', fontsize=8)
    
    # Bottom row: display frames
    ax2.axis('off')
    
    # Calculate grid layout
    cols = min(6, len(sampled_indices))
    rows = (len(sampled_indices) + cols - 1) // cols
    
    # Extract frames and display in grid
    for i, frame_idx in enumerate(sampled_indices):
        # Extract frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # Placeholder for missing frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate position in grid
        row, col = i // cols, i % cols
        
        # Create subplot
        sub_ax = ax2.inset_axes([col/cols, 1-(row+1)/rows, 1/cols, 1/rows])
        
        # Display frame
        sub_ax.imshow(frame)
        sub_ax.axis('off')
        sub_ax.set_title(f'Frame {frame_idx}\n({frame_idx/fps:.2f}s)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    cap.release()
    return True

def visualize_from_csv(csv_path, videos_root, output_dir, max_videos=-1):
    """
    Visualize sampling patterns from a CSV file
    
    Args:
        csv_path: Path to the CSV file
        videos_root: Root directory for videos
        output_dir: Output directory for visualizations
        max_videos: Maximum number of videos to visualize (-1 for all videos)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sampling method from CSV filename
    try:
        filename = os.path.basename(csv_path)
        parts = filename.split('_')
        if len(parts) >= 3:
            sampling_method = parts[3]  # Format: sampling_indices_dataset_method_split.csv
        else:
            sampling_method = "unknown"
    except:
        sampling_method = "unknown"
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Read CSV file with {len(df)} rows")
        
        if len(df) == 0:
            print("CSV file is empty!")
            return
            
        # Print CSV header to debug
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Print first row to debug
        if len(df) > 0:
            print(f"First row: {df.iloc[0].tolist()}")
        
        # Process rows
        processed_count = 0
        for idx, row in df.iterrows():
            if max_videos >= 0 and processed_count >= max_videos:
                print(f"Reached maximum number of videos to visualize ({max_videos})")
                break
                
            # Get video info
            video_filename = row['video_filename']
            total_frames = int(row['total_frames'])
            
            # Extract indices (all columns from the 3rd onwards)
            # Use index number 2 and beyond to handle any column name
            indices = row.iloc[2:].dropna().astype(int).tolist()
            
            if not indices:
                print(f"No valid indices found for {video_filename}, skipping")
                continue
                
            print(f"Processing {video_filename} with {len(indices)} indices")
            
            # Find video file
            video_path = find_video_file(videos_root, video_filename)
            if not video_path:
                print(f"Could not find video file for {video_filename}, skipping")
                continue
            
            # Create visualization
            output_path = os.path.join(output_dir, f"{video_filename}_{sampling_method}_sampling.png")
            if create_sampling_visualization(video_path, indices, sampling_method, 42, output_path):
                processed_count += 1
                print(f"Created visualization for {video_filename} ({processed_count}/{len(df) if max_videos < 0 else max_videos})")
            
        print(f"Created {processed_count} visualizations in {output_dir}")
        
    except Exception as e:
        import traceback
        print(f"Error processing CSV file {csv_path}: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sampling patterns from CSV files")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--videos_root", type=str, required=True, help="Root directory for videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--max_videos", type=int, default=-1, help="Maximum number of videos to visualize (-1 for all)")
    
    args = parser.parse_args()
    visualize_from_csv(args.csv, args.videos_root, args.output_dir, args.max_videos)
