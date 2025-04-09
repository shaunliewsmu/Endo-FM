#!/bin/bash
# scripts/train_foundation_phase1.sh
# Set CUDA device
# export CUDA_VISIBLE_DEVICES=1
# Configuration
EXP_NAME="foundation_phase1_second"
DATASET="ucf101"
DATA_PATH="data/downstream/duhs-gss-split-5:v0"
CHECKPOINT="checkpoints/endo_fm.pth"

# Create directories
mkdir -p logs/$EXP_NAME
mkdir -p models/$EXP_NAME

echo "Starting Phase 1: Fine-tuning Foundation Model on frames..."

python main_foundation_phase1.py \
  --data_dir "$DATA_PATH" \
  --log_dir "logs/$EXP_NAME" \
  --model_dir "models/$EXP_NAME" \
  --train_sampling "uniform" \
  --val_sampling "uniform" \
  --test_sampling "uniform" \
  --num_frames 8 \
  --batch_size 8 \
  --epochs 5 \
  --learning_rate 0.0001 \
  --arch "vit_base" \
  --patch_size 16 \
  --num_classes 2 \
  --pretrained_weights "$CHECKPOINT" \
  --loss_type "focal" \
  --focal_alpha 0.55 \
  --focal_gamma 2.0 \
  --patience 5 \
  --use_wandb \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False

echo "Phase 1 completed!"