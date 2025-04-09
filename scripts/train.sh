#!/bin/bash

# Script for training with custom sampling and focal loss - with fixed seed for reproducibility
EXP_NAME="endo-fm-duke-uniform-focal-25-15-training"
DATASET="ucf101"
DATA_PATH="data/downstream/duhs-gss-split-5:v0"
CHECKPOINT="checkpoints/endo_fm.pth"

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

# Apply dataset patch to ensure proper integration of SamplingDatasetMixin
python utils/datasets_patch.py

# number of frames to sample is 8 instead of 32 as the pretrained model are trained on 8 frames
# Run training with reproducible sampling (seed=42)
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_finetune_new.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 30 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 4 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --train_sampling "uniform" \
  --val_sampling "uniform" \
  --test_sampling "uniform" \
  --num_frames 8 \
  --loss_function "focal_loss" \
  --focal_gamma 2.5 \
  --focal_alpha 1.5 \
  --seed 42 \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Training completed successfully. Creating visualizations..."
  
  # Find the available CSV files
  SAMPLING_DIR="checkpoints/$EXP_NAME/sampling_indices"
  echo "Looking for CSV files in $SAMPLING_DIR"
  
  # List all available CSV files
  ls -la "$SAMPLING_DIR"
  
  # Try to find a CSV file to visualize
  CSV_FILE=""
  if [ -d "$SAMPLING_DIR" ]; then
    # First try to find the train random CSV
    CSV_FILE=$(find "$SAMPLING_DIR" -name "sampling_indices_${DATASET}_random_train.csv" -type f | head -n 1)
    
    # If not found, try any CSV
    if [ -z "$CSV_FILE" ]; then
      CSV_FILE=$(find "$SAMPLING_DIR" -name "*.csv" -type f | head -n 1)
    fi
  fi
  
  # Create sampling visualization dashboard
  # python utils/sampling_dashboard.py \
  #   --csv_dir "checkpoints/$EXP_NAME/sampling_indices" \
  #   --output_dir "checkpoints/$EXP_NAME/sampling_dashboard"

  # Visualize sample frames for a few videos (if CSV found)
  if [ -n "$CSV_FILE" ]; then
    echo "Using CSV file: $CSV_FILE for visualization"
    # python utils/visualize_sampling.py \
    #   --csv "$CSV_FILE" \
    #   --videos_root "${DATA_PATH}/videos" \
    #   --output_dir "checkpoints/$EXP_NAME/sampling_visualizations" \
    #   --max_videos 5
  else
    echo "No CSV files found for visualization. Check the sampling_indices directory."
  fi
    
  echo "Training and visualization completed with reproducible sampling (seed=42)!"
else
  echo "Training failed. Skipping visualizations."
  exit 1
fi