import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
from pathlib import Path

def visualize_sampling_from_csv(csv_path, videos_root, output_dir, max_videos=5):
    """
    Visualize the sampling patterns from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file with sampling indices
        videos_root (str): Root directory containing the videos
        output_dir (str): Directory to save visualizations
        max_videos (int): Maximum number of videos to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get sampling method from CSV filename
    sampling_method = os.path.basename(csv_path).split('_')[2]
    
    # Process up to max_videos
    for i, row in df.head(max_videos).iterrows():
        video_filename = row['video_filename']
        total_frames = row['total_frames']
        sampled_indices = [int(idx) for idx in row['sampled_indices'].split(',')]
        seed = row.get('seed', 42)  # Default to 42 if not specified
        
        # Find the video file
        video_path = find_video_file(videos_root, video_filename)
        if not video_path:
            print(f"Could not find video file for {video_filename}, skipping")
            continue
        
        # Visualize the sampling pattern
        try:
            visualization_path = os.path.join(output_dir, f"{video_filename}_{sampling_method}_sampling.png")
            create_sampling_visualization(video_path, sampled_indices, sampling_method, seed, visualization_path)
            print(f"Created visualization for {video_filename} at {visualization_path}")
        except Exception as e:
            print(f"Error creating visualization for {video_filename}: {str(e)}")

def find_video_file(root_dir, filename):
    """Search for a video file in the root directory."""
    for path in Path(root_dir).rglob(filename):
        return str(path)
    return None

def create_sampling_visualization(video_path, sampled_indices, sampling_method, seed, output_path):
    """Create a visualization of the sampled frames from a video."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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

def main():
    parser = argparse.ArgumentParser(description='Visualize sampling patterns from CSV files')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with sampling indices')
    parser.add_argument('--videos_root', type=str, required=True, help='Root directory containing the videos')
    parser.add_argument('--output_dir', type=str, default='sampling_visualizations', help='Directory to save visualizations')
    parser.add_argument('--max_videos', type=int, default=5, help='Maximum number of videos to visualize')
    
    args = parser.parse_args()
    visualize_sampling_from_csv(args.csv, args.videos_root, args.output_dir, args.max_videos)

if __name__ == '__main__':
    main()