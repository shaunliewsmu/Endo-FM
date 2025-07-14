#!/bin/bash

# Script for training with custom sampling, focal loss, and data augmentation - with fixed seed for reproducibility
EXP_NAME="augmented_new-new-bagls-fine-tune-balanced-uniform-focal-1-05-training-8"
DATASET="ucf101"
DATA_PATH="data/downstream/balanced-dataset"
# DATA_PATH="data/downstream/bagls-split:v0"
CHECKPOINT="checkpoints/augmented-new-bagls-balanced-uniform-focal-1-05-training-8/checkpoint.pth.tar"
CHECKPOINT="checkpoints/endo_fm.pth"
if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

# Run training with reproducible sampling (seed=42) and data augmentation
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_finetune_new.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 30 \
  --lr 0.001 \
  --batch_size_per_gpu 2 \
  --num_workers 2 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --train_sampling "uniform" \
  --val_sampling "uniform" \
  --test_sampling "uniform" \
  --num_frames 8 \
  --loss_function "focal_loss" \
  --focal_gamma 1 \
  --focal_alpha 0.5 \
  --scratch \
  --seed 42 \
  --augment \
  --aug_step_size 16 \
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
  
  # Create sampling visualization dashboard
  python utils/sampling_dashboard.py \
    --csv_dir "checkpoints/$EXP_NAME/sampling_indices" \
    --output_dir "checkpoints/$EXP_NAME/sampling_dashboard"

  # Visualize sample frames for a few videos
  CSV_FILE=$(find "$SAMPLING_DIR" -name "sampling_indices_${DATASET}_uniform_train_augstep2.csv" -type f | head -n 1)
  if [ -n "$CSV_FILE" ]; then
    echo "Using augmented CSV file: $CSV_FILE for visualization"
    python utils/visualize_sampling.py \
      --csv "$CSV_FILE" \
      --videos_root "${DATA_PATH}/videos" \
      --output_dir "checkpoints/$EXP_NAME/sampling_visualizations" \
      --max_videos 5
  else
    echo "No augmented CSV files found. Checking for regular files..."
    CSV_FILE=$(find "$SAMPLING_DIR" -name "*.csv" -type f | head -n 1)
    if [ -n "$CSV_FILE" ]; then
      echo "Using CSV file: $CSV_FILE for visualization"
      python utils/visualize_sampling.py \
        --csv "$CSV_FILE" \
        --videos_root "${DATA_PATH}/videos" \
        --output_dir "checkpoints/$EXP_NAME/sampling_visualizations" \
        --max_videos 5
    else
      echo "No CSV files found for visualization. Check the sampling_indices directory."
    fi
  fi
    
  echo "Training and visualization completed with reproducible sampling (seed=42) and data augmentation!"
else
  echo "Training failed. Skipping visualizations."
  exit 1
fi