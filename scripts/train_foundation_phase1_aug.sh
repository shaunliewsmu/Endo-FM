#!/bin/bash
# scripts/train_foundation_phase1.sh
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
# Configuration
EXP_NAME="augmented_foundation_phase1_uniform_fine_tune"  # Update experiment name
DATASET="ucf101"
DATA_PATH="data/downstream/balanced-dataset"
# DATA_PATH="data/downstream/bagls-split:v0"
# CHECKPOINT="checkpoints/endo_fm.pth"  
CHECKPOINT="models/foundation_phase1_uniform_bagls/foundation_model_20250427_175740.pth"

# Create directories
if [ ! -d "logs/$EXP_NAME" ]; then
  mkdir -p "logs/$EXP_NAME"
fi
if [ ! -d "models/$EXP_NAME" ]; then
  mkdir -p "models/$EXP_NAME"
fi

echo "Starting Phase 1: Fine-tuning Foundation Model on frames with data augmentation..."

# Use distributed training with augmentation parameters
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  main_foundation_phase1.py \
  --data_dir "$DATA_PATH" \
  --log_dir "logs/$EXP_NAME" \
  --model_dir "models/$EXP_NAME" \
  --train_sampling "uniform" \
  --val_sampling "uniform" \
  --test_sampling "uniform" \
  --num_frames 32 \
  --batch_size 2 \
  --num_workers 0 \
  --epochs 32 \
  --learning_rate 0.0001 \
  --arch "vit_base" \
  --patch_size 8 \
  --num_classes 2 \
  --pretrained_weights "$CHECKPOINT" \
  --loss_type "focal" \
  --focal_alpha 0.55 \
  --focal_gamma 2.0 \
  --patience 10 \
  --seed 42 \
  --scratch \
  --augment \
  --aug_step_size 16 \
  --use_wandb \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Training completed successfully."
  
  # Check for sampling indices directory
  SAMPLING_DIR="logs/$EXP_NAME/sampling_indices"
  if [ -d "$SAMPLING_DIR" ]; then
    echo "Looking for CSV files in $SAMPLING_DIR"
    # List all available CSV files
    ls -la "$SAMPLING_DIR"
    
    # Create sampling visualization dashboard
    if [ -n "$(ls -A $SAMPLING_DIR/*.csv 2>/dev/null)" ]; then
      python utils/sampling_dashboard.py \
        --csv_dir "$SAMPLING_DIR" \
        --output_dir "logs/$EXP_NAME/sampling_dashboard"
        
      # Visualize sample frames
      CSV_FILE=$(find "$SAMPLING_DIR" -name "sampling_indices_${DATASET}_uniform_train_augstep16.csv" -type f | head -n 1)
      if [ -n "$CSV_FILE" ]; then
        echo "Using augmented CSV file: $CSV_FILE for visualization"
        python utils/visualize_sampling.py \
          --csv "$CSV_FILE" \
          --videos_root "${DATA_PATH}/videos" \
          --output_dir "logs/$EXP_NAME/sampling_visualizations" \
          --max_videos 5
      fi
    fi
  fi
  
  echo "Phase 1 completed successfully!"
else
  echo "Training failed."
  exit 1
fi