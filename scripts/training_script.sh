#!/bin/bash

# This script directly adds CSV recording functionality to the training process
# without requiring complex patching of dataset classes

# Configuration
EXP_NAME="endo-fm-duke-uniform-focal-25-15-result"
DATASET="ucf101"
DATA_PATH="data/downstream/duhs-gss-split-5:v0"
CHECKPOINT="checkpoints/endo_fm.pth"
SAMPLING_DIR="checkpoints/$EXP_NAME/sampling_indices"
TRAIN_SAMPLING="uniform"
VAL_SAMPLING="uniform"
TEST_SAMPLING="uniform"
# Create directory for sampling indices
mkdir -p "$SAMPLING_DIR"

# Run the regular training script
echo "Running training script..."
bash scripts/train.sh

# After training, manually create the CSV files expected by visualization
echo "Creating CSV files for visualization..."

# Generate CSV files using actual dataset videos
python generate_csv.py \
  --output_dir "$SAMPLING_DIR" \
  --dataset "$DATASET" \
  --train_sampling "$TRAIN_SAMPLING" \
  --val_sampling "$VAL_SAMPLING" \
  --test_sampling "$TEST_SAMPLING" \
  --video_root "${DATA_PATH}/videos" \
  --splits_path "${DATA_PATH}/splits" \
  --num_frames 8

# Check if CSV files were created
echo "Checking for CSV files in $SAMPLING_DIR"
ls -la "$SAMPLING_DIR"

# Try to visualize the sampling patterns for all videos
if [ -d "$SAMPLING_DIR" ]; then
    # Check train CSV
    TRAIN_CSV="$SAMPLING_DIR/sampling_indices_${DATASET}_${TRAIN_SAMPLING}_train.csv"
    if [ -f "$TRAIN_CSV" ]; then
        echo "Processing training set CSV: $TRAIN_CSV"
        
        # # Create visualization directory
        # mkdir -p "checkpoints/$EXP_NAME/sampling_visualizations/train"
        
        # Run visualization with our custom script
        # python custom_visualize.py \
        #     --csv "$TRAIN_CSV" \
        #     --videos_root "${DATA_PATH}/videos" \
        #     --output_dir "checkpoints/$EXP_NAME/sampling_visualizations/train" \
        #     --max_videos -1
            
        # echo "Created train visualizations in checkpoints/$EXP_NAME/sampling_visualizations/train"
    fi
    
    # Check val CSV
    VAL_CSV="$SAMPLING_DIR/sampling_indices_${DATASET}_${VAL_SAMPLING}_val.csv"
    if [ -f "$VAL_CSV" ]; then
        echo "Processing validation set CSV: $VAL_CSV"
        
        # # Create visualization directory
        # mkdir -p "checkpoints/$EXP_NAME/sampling_visualizations/val"
        
        # Run visualization with our custom script
        # python custom_visualize.py \
        #     --csv "$VAL_CSV" \
        #     --videos_root "${DATA_PATH}/videos" \
        #     --output_dir "checkpoints/$EXP_NAME/sampling_visualizations/val" \
        #     --max_videos -1
            
        # echo "Created val visualizations in checkpoints/$EXP_NAME/sampling_visualizations/val"
    fi
else
    echo "No sampling indices directory found."
fi

echo "Process completed!"