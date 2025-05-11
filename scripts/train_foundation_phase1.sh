#!/bin/bash
# scripts/train_foundation_phase1.sh
# Set CUDA device
export CUDA_VISIBLE_DEVICES=1
# Configuration
EXP_NAME="foundation_phase1_uniform_fine_tune"
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

# Apply dataset patch to ensure proper integration of SamplingDatasetMixin (if needed)
if [ -f "utils/datasets_patch.py" ]; then
  echo "Applying dataset patch..."
  python utils/datasets_patch.py
fi

echo "Starting Phase 1: Fine-tuning Foundation Model on frames..."

# Use distributed training like in the working script
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
  --use_wandb \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Training completed successfully."

  # Check for sampling indices directory if your code creates it
  SAMPLING_DIR="logs/$EXP_NAME/sampling_indices"
  if [ -d "$SAMPLING_DIR" ]; then
    echo "Looking for CSV files in $SAMPLING_DIR"
    # List all available CSV files
    ls -la "$SAMPLING_DIR"
    
    # If you have visualization tools, you can uncomment these lines
    # CSV_FILE=$(find "$SAMPLING_DIR" -name "*.csv" -type f | head -n 1)
    # if [ -n "$CSV_FILE" ]; then
    #   echo "Using CSV file: $CSV_FILE for visualization"
    #   python utils/visualize_sampling.py \
    #     --csv "$CSV_FILE" \
    #     --videos_root "${DATA_PATH}/videos" \
    #     --output_dir "logs/$EXP_NAME/sampling_visualizations" \
    #     --max_videos 5
    # fi
  fi
  
  echo "Phase 1 completed successfully!"
else
  echo "Training failed."
  exit 1
fi