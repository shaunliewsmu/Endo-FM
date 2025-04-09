#!/bin/bash
# scripts/train_foundation_lstm.sh

# Configuration
DATASET="ucf101"
DATA_PATH="data/downstream/duhs-gss-split-5:v0"
CHECKPOINT="checkpoints/endo_fm.pth"

# Run Phase 1 (optional)
echo "Running Phase 1: Foundation Model fine-tuning..."
PHASE1_OUTPUT=$(bash scripts/train_foundation_phase1.sh)
echo "$PHASE1_OUTPUT"

# Extract the model path from Phase 1 output
FINE_TUNED_MODEL=$(echo "$PHASE1_OUTPUT" | grep -o "Saved best model to models/foundation_phase1/foundation_model_.*\.pth" | awk '{print $5}')

if [ -n "$FINE_TUNED_MODEL" ]; then
  echo "Using fine-tuned foundation model from Phase 1: $FINE_TUNED_MODEL"
  # Update the checkpoint in the Phase 2 script with the fine-tuned model
  sed -i "s|FOUNDATION_CHECKPOINT=.*|FOUNDATION_CHECKPOINT=\"$FINE_TUNED_MODEL\"|" scripts/train_foundation_phase2.sh
  bash scripts/train_foundation_phase2.sh
else
  echo "Using pre-trained foundation model: $CHECKPOINT"
  # Run Phase 2 with the original foundation model
  bash scripts/train_foundation_phase2.sh
fi

echo "Foundation Model + LSTM two-phase training completed!"