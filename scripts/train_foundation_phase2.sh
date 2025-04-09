#!/bin/bash
# scripts/train_foundation_phase2.sh
# Set CUDA device
export CUDA_VISIBLE_DEVICES=1
# Configuration
EXP_NAME="foundation_phase2"
DATASET="ucf101"
DATA_PATH="data/downstream/duhs-gss-split-5:v0"
# FOUNDATION_CHECKPOINT="checkpoints/endo_fm.pth"  # Use foundation model directly
FOUNDATION_CHECKPOINT="models/foundation_phase1/foundation_model_20250409_102656.pth"  # Use fine-tuned model if available
FEATURES_DIR="features/foundation_model"

# Create directories
mkdir -p logs/$EXP_NAME
mkdir -p models/$EXP_NAME
mkdir -p $FEATURES_DIR

echo "Starting Phase 2: Foundation Model + LSTM..."

python main_foundation_phase2.py \
  --data_dir "$DATA_PATH" \
  --features_dir "$FEATURES_DIR" \
  --log_dir "logs/$EXP_NAME" \
  --model_dir "models/$EXP_NAME" \
  --foundation_checkpoint "$FOUNDATION_CHECKPOINT" \
  --extract_features \
  --sampling_method "uniform" \
  --num_frames 8 \
  --input_size 768 \
  --hidden_size 512 \
  --num_layers 2 \
  --dropout 0.5 \
  --bidirectional \
  --batch_size 32 \
  --epochs 30 \
  --learning_rate 0.001 \
  --num_classes 2 \
  --loss_type "focal" \
  --focal_alpha 0.55 \
  --focal_gamma 2.0 \
  --patience 7 \
  --use_wandb \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False

echo "Phase 2 completed!"