# Endo-FM Video Classifier

This is an Endo-FM (Endoscopy Foundation Model) implementation for laryngeal cancer screening that utilizes pre-trained foundation models for fine-tuning on downstream video classification tasks. Endo-FM is implemented in a **separate repository** and uses a distributed training framework optimized for medical endoscopy videos.

## Project Overview

This Endo-FM implementation offers:
- **Foundation model approach**: Leverages pre-trained Endo-FM models for downstream fine-tuning
- **Medical video specialization**: Specifically designed for endoscopy video understanding
- **Distributed training**: Multi-GPU support with PyTorch distributed training
- **Comprehensive visualization**: Automatic sampling indices, confusion matrices, and training dashboards
- **Reproducible training**: Fixed seed (42) for consistent results across runs

## Repository Setup

### Installation

```bash
git clone https://github.com/mhleerepo/Endo-FM.git  
cd Endo-FM
conda env create -f environment.yaml
conda activate endofm
```

> **Note**: If running on mercury server, the conda environment already exists. Simply run:
> ```bash
> conda activate endofm
> ```

### Dataset Installation

All preprocessed datasets are available on the mercury server. Copy the datasets to your cloned repository:

#### Available Datasets
1. **Balanced (Duke + BAGLS) Dataset** (`data/downstream/balanced-dataset`)
2. **BAGLS Dataset** (`data/downstream/bagls-split:v0`)
3. **Duke Dataset** (`data/downstream/duhs-gss-split-5:v0`)
4. **PolypDiag Dataset** (`data/downstream/PolypDiag`) - Default dataset provided by Endo-FM

#### Copy Commands

SSH into the mercury server and navigate to `/mnt/storage/shaun/Endo-FM/data/downstream`, then copy:

```bash
# Copy downstream data folder
rsync -av /mnt/storage/shaun/Endo-FM/data/downstream/ /path/to/destination/Endo-FM/data/downstream/
# OR
cp -r /mnt/storage/shaun/Endo-FM/data/downstream/ /path/to/destination/Endo-FM/data/downstream/
```

#### Pre-trained Model Download

Download the pre-trained Endo-FM model from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EZh5mWE5CL1BpaJ1bXuokfYBDM2VaMknqG7YpaQBRgAvdQ?e=e2rVYW) and place it under `checkpoints/`:
- Model file: `endo_fm.pth`
- Location: `checkpoints/endo_fm.pth`

## Project Structure

```
Endo-FM/
├── README.md                                      # Project documentation
├── environment.yaml                               # Conda environment configuration
├── custom_visualize.py                            # Custom visualization script
├── generate_csv.py                                # CSV generation utility
├── visualize_csv.py                               # CSV visualization script
├── checkpoints/                                   # Training outputs and checkpoints
│   ├── endo_fm.pth                               # Pre-trained Endo-FM model
│   ├── augmented-new-bagls-balanced-random-focal-1-05-training-8/
│   │   ├── best_confusion_matrix_epoch_0.png
│   │   ├── checkpoint.pth.tar                    # Trained model checkpoint
│   │   ├── config.json                           # Configuration used for training
│   │   ├── training.log                          # Training logs
│   │   ├── sampling_indices/                     # Sampling index files (CSV)
│   │   │   ├── sampling_indices_ucf101_random_window_train_augstep16.csv
│   │   │   └── sampling_indices_ucf101_random_window_val.csv
│   │   └── sampling_dashboard/                   # Sampling dashboard visualizations
│   └── ...                                       # Additional training folders
├── data/                                          # Data storage
│   └── downstream/
│       ├── bagls-split:v0/                       # BAGLS dataset
│       ├── balanced-dataset/                     # Balanced Duke + BAGLS dataset
│       ├── duhs-gss-split-5:v0/                  # Duke dataset
│       └── PolypDiag/                            # PolypDiag dataset
├── datasets/                                      # Dataset handling scripts
│   └── ...                                       # Various dataset utilities
├── scripts/                                       # Training scripts
│   ├── train_8frames.sh                          # Main training script with 8 frames
│   ├── train_aug_8frames.sh                      # Augmented training script with 8 frames
│   ├── train.sh                                  # Original training script (32 frames)
│   ├── train_aug.sh                              # Original augmented training script (32 frames)
│   └── ...                                       # Additional utility scripts
├── utils/                                         # Utility functions and helpers
│   ├── datasets_patch.py                         # Dataset integration patch
│   ├── sampling_dashboard.py                     # Sampling visualization dashboard
│   ├── visualize_sampling.py                     # Sampling visualization utility
│   └── ...                                       # Various utility scripts
├── main_foundation_phase1.py                      # Foundation model training script phase 1
├── main_foundation_phase2.py                      # Foundation model training script phase 2
├── eval_finetune_new.py                           # Main evaluation and fine-tuning script
└── ...                                            # Additional configuration files
```

## Training Output Structure

Each training run creates a directory under `checkpoints/`:

```
[EXP_NAME]/
├── best_confusion_matrix_epoch_0.png              # Best confusion matrix visualization
├── best_confusion_matrix_epoch_1.png              # Confusion matrices for each epoch
├── checkpoint.pth.tar                            # Model checkpoint
├── config.json                                   # Configuration used for the run
├── log.txt                                       # Training log file
├── training.log                                  # Detailed training logs
├── sampling_indices/                             # Sampling index files (CSV)
│   ├── sampling_indices_ucf101_[method]_train.csv
│   ├── sampling_indices_ucf101_[method]_train_augstep16.csv
│   └── sampling_indices_ucf101_[method]_val.csv
└── sampling_dashboard/                           # Sampling dashboard visualizations
```

## Training Methods

We have **4 training methods** available for Endo-FM:

### 1. Without Data Augmentation and Without Fine Tune

Direct training on balanced dataset using the pre-trained Endo-FM model.

**Script Location**: `scripts/train_8frames.sh`

**Key Variables to Configure**:
```bash
EXP_NAME="new-endo-fm-balanced-fine-tune-random-focal-1-05-training-8"
DATASET="ucf101"                                  # Must remain "ucf101"
DATA_PATH="data/downstream/balanced-dataset"     # Dataset path
CHECKPOINT="checkpoints/endo_fm.pth"             # Pre-trained model

# Training Parameters
--epochs 30                                       # Number of epochs
--train_sampling "random"                        # Sampling method: random, random_window, uniform
--val_sampling "random"                          # Validation sampling
--test_sampling "random"                         # Test sampling
--num_frames 8                                   # Number of frames (optimized for Endo-FM)
--focal_gamma 1                                  # Focal loss gamma
--focal_alpha 0.5                                # Focal loss alpha
```

**Command to Run**:
```bash
bash scripts/train_8frames.sh
```

### 2. Without Data Augmentation and With Fine Tune

Two-stage training: first train on BAGLS dataset, then fine-tune on balanced dataset.

#### First Training (BAGLS Dataset):

**Configure `scripts/train_8frames.sh`**:
```bash
EXP_NAME="endo-fm-bagls-first-training"
DATA_PATH="data/downstream/bagls-split:v0"
CHECKPOINT="checkpoints/endo_fm.pth"
```

**Command to Run**:
```bash
bash scripts/train_8frames.sh
```

#### Second Training (Fine-tuning on Balanced Dataset):

**Configure `scripts/train_8frames.sh`**:
```bash
EXP_NAME="endo-fm-balanced-fine-tune"
DATA_PATH="data/downstream/balanced-dataset"
CHECKPOINT="checkpoints/endo-fm-bagls-first-training/checkpoint.pth.tar"  # From first training
```

**Command to Run**:
```bash
bash scripts/train_8frames.sh
```

> **Important**: Update the `CHECKPOINT` path to point to the checkpoint generated from the first training.

### 3. With Data Augmentation and Without Fine Tune

Direct training on balanced dataset with data augmentation enabled.

**Script Location**: `scripts/train_aug_8frames.sh`

**Key Variables to Configure**:
```bash
EXP_NAME="augmented-endo-fm-balanced-training"
DATASET="ucf101"
DATA_PATH="data/downstream/balanced-dataset"
CHECKPOINT="checkpoints/endo_fm.pth"

# Augmentation Parameters
--augment                                         # Enable data augmentation
--aug_step_size 16                               # Optimal augmentation step size
--scratch                                        # Training from scratch flag
```

**Command to Run**:
```bash
bash scripts/train_aug_8frames.sh
```

### 4. With Data Augmentation and With Fine Tune

Two-stage training with data augmentation in both stages.

#### First Training (BAGLS Dataset with Augmentation):

**Configure `scripts/train_aug_8frames.sh`**:
```bash
EXP_NAME="augmented-endo-fm-bagls-first"
DATA_PATH="data/downstream/bagls-split:v0"
CHECKPOINT="checkpoints/endo_fm.pth"
--augment
--aug_step_size 16
```

**Command to Run**:
```bash
bash scripts/train_aug_8frames.sh
```

#### Second Training (Fine-tuning with Augmentation):

**Configure `scripts/train_aug_8frames.sh`**:
```bash
EXP_NAME="augmented-endo-fm-balanced-fine-tune"
DATA_PATH="data/downstream/balanced-dataset"
CHECKPOINT="checkpoints/augmented-endo-fm-bagls-first/checkpoint.pth.tar"  # From first training
--augment
--aug_step_size 16
```

**Command to Run**:
```bash
bash scripts/train_aug_8frames.sh
```

## Parameter Customization

### Core Training Parameters
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size_per_gpu`: Batch size per GPU (default: 2)
- `--num_workers`: Number of data loading workers (default: 2)
- `--num_labels`: Number of classification labels (default: 2)
- `--num_frames`: Number of frames to sample (default: 8, optimized for Endo-FM)

### Sampling Methods and Data Augmentation Theory
you can find more details about the theory used for sampling methods and data augmentation from [here](https://github.com/mhleerepo/ai-laryngeal-video-based-classifier/blob/main/README_Techniques_Explaination.md)

### Sampling Methods
Available sampling methods for `--train_sampling`, `--val_sampling`, and `--test_sampling`:
- **random**: Randomly sample frames
- **random_window**: Random sampling within windows
- **uniform**: Uniformly sample frames across video

### Loss Function Parameters
- `--loss_function`: Loss function type (default: "focal_loss")
- `--focal_gamma`: Focal loss gamma parameter (default: 1)
- `--focal_alpha`: Focal loss alpha parameter (default: 0.5)

### Data Augmentation Parameters
- `--augment`: Enable data augmentation
- `--aug_step_size`: Control augmentation rounds (optimal: 16)
- `--scratch`: Training from scratch flag (used with augmentation)

### Model Architecture Parameters
- `--arch`: Model architecture (default: "vit_base")
- `--n_last_blocks`: Number of last blocks to fine-tune (default: 1)

### Reproducibility Parameters
- `--seed`: Random seed for reproducible results (default: 42)

### Dataset Configuration
- `DATA.PATH_TO_DATA_DIR`: Path to data directory splits
- `DATA.PATH_PREFIX`: Path to video files
- `DATA.USE_FLOW`: Use optical flow (default: False)

## Model Features

### Endo-FM Architecture Advantages
- **Foundation Model Approach**: Leverages pre-trained models specialized for endoscopy
- **Medical Video Understanding**: Optimized for medical endoscopy video classification
- **Vision Transformer Backbone**: Uses ViT-Base architecture for robust feature extraction
- **Transfer Learning**: Effective knowledge transfer from pre-trained Endo-FM to downstream tasks
- **Efficient Training**: 8-frame sampling optimized for the pre-trained model

### Training Framework Features
- **Distributed Training**: PyTorch distributed training with automatic port assignment
- **Automatic Visualization**: Generated sampling indices, confusion matrices, and dashboards
- **Reproducible Results**: Fixed seed (42) ensures consistent training outcomes
- **Comprehensive Logging**: Detailed training logs and configuration tracking
- **Dataset Integration**: Automatic dataset patching for proper integration

## Important Notes

### Environment Requirements
- **Conda Environment**: Uses conda instead of pip for package management
- **Python Distributed**: Requires PyTorch distributed training setup
- **GPU Requirements**: Optimized for single-GPU training (`--nproc_per_node=1`)

### Dataset Requirements
- **Fixed Dataset Name**: `DATASET` must remain "ucf101" for compatibility
- **Directory Structure**: Requires specific `splits/` and `videos/` subdirectories
- **Frame Count**: Optimized for 8 frames instead of 32 (pre-trained model requirement)

### Checkpoint Management
- **Pre-trained Model**: Must download and place `endo_fm.pth` in `checkpoints/`
- **Fine-tuning Checkpoints**: Use `checkpoint.pth.tar` from previous training runs
- **Experiment Naming**: Use descriptive `EXP_NAME` for easy identification of training runs

### Visualization Features
- **Automatic Generation**: Confusion matrices, sampling indices, and dashboards created automatically
- **CSV Tracking**: Detailed sampling indices saved as CSV files for analysis
- **Dashboard Creation**: Interactive sampling visualization dashboards
- **Frame Visualization**: Sample frame visualizations for understanding model behavior

### Training Process
- **Dataset Patching**: Automatic application of dataset patches before training
- **Port Assignment**: Random port assignment for distributed training
- **Success Verification**: Automatic success checking and visualization triggering
- **Log Management**: Comprehensive logging with training progress and metrics

### Best Practices
- **Experiment Naming**: Use descriptive names for `EXP_NAME` to organize training runs
- **Checkpoint Paths**: Always verify correct checkpoint paths when fine-tuning
- **Resource Management**: Monitor GPU memory usage with 8-frame processing
- **Augmentation Control**: Use `aug_step_size=16` as the optimal setting for computational efficiency
- **Reproducibility**: Keep `--seed 42` for consistent results across experiments

### Training Results Location
All training results are automatically saved to:
`checkpoints/[EXP_NAME]/`

Each run creates a complete training environment with checkpoints, logs, visualizations, and sampling analysis.