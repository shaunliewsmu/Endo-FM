import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import seaborn as sns

def create_sampling_dashboard(csv_dir, output_dir):
    """
    Create a dashboard to compare different sampling methods.
    
    Args:
        csv_dir (str): Directory containing CSV files with sampling indices
        output_dir (str): Directory to save dashboard visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, 'sampling_indices_*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        
        # Create an empty dashboard to indicate no data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No sampling data available.\nCSV files were not created during training.", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'no_data_available.png'))
        plt.close()
        return
    
    # Group CSV files by dataset and split
    grouped_files = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) >= 4:
            dataset = parts[2]
            method = parts[3]
            split = parts[4] if len(parts) > 4 else 'unknown'
            
            key = f"{dataset}_{split}"
            if key not in grouped_files:
                grouped_files[key] = []
            
            grouped_files[key].append((method, csv_file))
    
    # Process each dataset and split
    for key, files in grouped_files.items():
        create_comparison_plots(key, files, output_dir)

def create_comparison_plots(key, files, output_dir):
    """Create comparison plots for different sampling methods for a dataset and split."""
    dataset, split = key.split('_')
    
    # Sampling range distribution plots
    plt.figure(figsize=(12, 6))
    
    for method, csv_file in files:
        df = pd.read_csv(csv_file)
        
        # Calculate normalized frame positions
        all_positions = []
        for _, row in df.iterrows():
            indices = [int(idx) for idx in row['sampled_indices'].split(',')]
            total_frames = row['total_frames']
            # Normalize to [0, 1] range
            normalized = [idx / total_frames for idx in indices]
            all_positions.extend(normalized)
        
        # Plot histogram
        sns.kdeplot(all_positions, label=f"{method.capitalize()}", fill=True, alpha=0.3)
    
    plt.title(f"Frame Position Distribution - {dataset.upper()} ({split})\nUsing seed=42 for reproducibility")
    plt.xlabel('Normalized Frame Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{dataset}_{split}_position_distribution.png"), dpi=150)
    plt.close()
    
    # Frame coverage plot
    plt.figure(figsize=(12, 6))
    
    for method, csv_file in files:
        df = pd.read_csv(csv_file)
        
        # Calculate average coverage
        coverages = []
        for _, row in df.iterrows():
            indices = [int(idx) for idx in row['sampled_indices'].split(',')]
            total_frames = row['total_frames']
            
            # Calculate coverage (how evenly distributed are the frames)
            if len(indices) > 1 and total_frames > 1:
                # Normalize indices to [0, 1]
                norm_indices = [idx / total_frames for idx in indices]
                # Sort them
                norm_indices.sort()
                # Calculate gaps between consecutive indices
                gaps = [norm_indices[i+1] - norm_indices[i] for i in range(len(norm_indices)-1)]
                # Coefficient of variation of gaps (lower means more uniform coverage)
                cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                coverages.append(cv)
        
        if coverages:
            plt.boxplot(coverages, positions=[files.index((method, csv_file))], 
                      widths=0.6, labels=[method.capitalize()])
    
    plt.title(f"Frame Coverage Uniformity - {dataset.upper()} ({split})\nUsing seed=42 for reproducibility")
    plt.xlabel('Sampling Method')
    plt.ylabel('Coefficient of Variation (lower is more uniform)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{dataset}_{split}_coverage_uniformity.png"), dpi=150)
    plt.close()
    
    # Number of sampled frames per video
    plt.figure(figsize=(12, 6))
    
    for method, csv_file in files:
        df = pd.read_csv(csv_file)
        
        # Count frames per video
        frames_per_video = []
        for _, row in df.iterrows():
            indices = row['sampled_indices'].split(',')
            frames_per_video.append(len(indices))
        
        plt.hist(frames_per_video, alpha=0.7, label=f"{method.capitalize()}")
    
    plt.title(f"Frames Per Video - {dataset.upper()} ({split})\nUsing seed=42 for reproducibility")
    plt.xlabel('Number of Frames')
    plt.ylabel('Number of Videos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{dataset}_{split}_frames_per_video.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create sampling method comparison dashboard')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing CSV files with sampling indices')
    parser.add_argument('--output_dir', type=str, default='sampling_dashboard', help='Directory to save dashboard visualizations')
    
    args = parser.parse_args()
    create_sampling_dashboard(args.csv_dir, args.output_dir)

if __name__ == '__main__':
    main()