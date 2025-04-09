import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from pathlib import Path
import logging
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import get_vit_base_patch16_224
from datasets import UCF101
from utils import utils
from utils.parser import load_config
from utils.focal_loss import FocalLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, balanced_accuracy_score

def evaluate_lstm_on_test_set(model, test_loader, criterion, args, logger, device):
    """Evaluate LSTM model on test set with comprehensive metrics"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_all_preds = []
    test_all_labels = []
    test_all_probs = []
    
    logger.info("Evaluating LSTM model on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            try:
                # Handle individual feature vectors
                if len(inputs.shape) == 2:
                    if inputs.shape[0] == test_loader.batch_size:
                        # Shape is [batch_size, feature_dim]
                        inputs = inputs.unsqueeze(1)  # Reshape to [batch_size, 1, feature_dim]
                
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update statistics
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
                
                # Store predictions and labels for metrics
                test_all_preds.extend(predicted.cpu().numpy())
                test_all_labels.extend(labels.cpu().numpy())
                
                # For binary classification, store probabilities
                if args.num_classes == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    test_all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in test batch: {str(e)}")
                continue
    
    # Calculate metrics
    test_loss = test_loss / test_total if test_total > 0 else float('inf')
    test_accuracy = test_correct / test_total if test_total > 0 else 0
    
    # Convert to numpy arrays
    test_all_preds = np.array(test_all_preds)
    test_all_labels = np.array(test_all_labels)
    
    # Calculate precision, recall, f1
    test_precision = precision_score(test_all_labels, test_all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_all_labels, test_all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_all_labels, test_all_preds, average='weighted', zero_division=0)
    test_balanced_accuracy = balanced_accuracy_score(test_all_labels, test_all_preds)
    
    # Calculate confusion matrix
    test_cm = confusion_matrix(test_all_labels, test_all_preds)
    
    # Binary classification metrics
    test_auroc = 0.5  # Default
    test_auprc = 0.0  # Default
    test_specificity = 0.0  # Default
    
    if args.num_classes == 2 and len(test_all_probs) > 0 and len(np.unique(test_all_labels)) > 1:
        # AUROC
        test_auroc = roc_auc_score(test_all_labels, test_all_probs)
        
        # AUPRC
        test_auprc = average_precision_score(test_all_labels, test_all_probs)
        
        # Specificity (true negative rate)
        tn, fp, fn, tp = test_cm.ravel()
        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Log all metrics
    logger.info("Test Metrics:")
    logger.info(f"Accuracy: {test_accuracy:.4f}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall: {test_recall:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    logger.info(f"AUROC: {test_auroc:.4f}")
    logger.info(f"PR-AUC: {test_auprc:.4f}")
    logger.info(f"Specificity: {test_specificity:.4f}")
    logger.info(f"Balanced Accuracy: {test_balanced_accuracy:.4f}")
    logger.info(f"Confusion Matrix: {test_cm.tolist()}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(args.num_classes),
                yticklabels=range(args.num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(os.path.join(args.log_dir, 'test_confusion_matrix_lstm.png'))
    plt.close()
    
    # For binary classification, also create ROC and PR curves
    if args.num_classes == 2 and len(test_all_probs) > 0:
        # ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(test_all_labels, test_all_probs)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {test_auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test Set ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(args.log_dir, 'test_roc_curve_lstm.png'))
        plt.close()
        
        # PR Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(test_all_labels, test_all_probs)
        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {test_auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Set Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(args.log_dir, 'test_pr_curve_lstm.png'))
        plt.close()
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auroc': test_auroc,
        'auprc': test_auprc,
        'specificity': test_specificity,
        'balanced_accuracy': test_balanced_accuracy,
        'confusion_matrix': test_cm
    }

# Create LSTM model
class FoundationLSTM(torch.nn.Module):
    """LSTM model for processing features from foundation model"""
    def __init__(self, input_size=768, hidden_size=512, num_layers=2, 
                 dropout=0.5, bidirectional=True, num_classes=2):
        super(FoundationLSTM, self).__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(lstm_out_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes)
        )
        
        # Add input normalization
        self.input_norm = torch.nn.LayerNorm(input_size)
    
    def forward(self, x):
        # Handle various input shapes
        # If x is just 2D [batch_size, features], reshape it to [batch_size, 1, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Normalize input features
        batch_size, seq_len, feat_dim = x.shape
        x = x.reshape(batch_size * seq_len, feat_dim)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, feat_dim)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use final time step
        final_hidden_state = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(final_hidden_state)
        
        return logits

class FeatureDataset(torch.utils.data.Dataset):
    """Dataset for loading extracted features"""
    def __init__(self, features_dir, mode='train', logger=None, input_size=768):
        self.features_dir = Path(features_dir) / mode
        self.mode = mode
        self.logger = logger
        self.input_size = input_size
        
        # Find all feature files
        self.feature_files = sorted(list(self.features_dir.glob('*.npy')))
        
        if not self.feature_files:
            if logger:
                logger.warning(f"No feature files found in {self.features_dir}")
        else:
            if logger:
                logger.info(f"Found {len(self.feature_files)} feature files in {mode} split")
        
        # Extract labels from filenames (format: index_label.npy)
        self.labels = []
        valid_files = []
        for file_path in self.feature_files:
            try:
                # Extract label from filename
                label = int(file_path.stem.split('_')[-1])
                self.labels.append(label)
                valid_files.append(file_path)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to parse label from {file_path.name}: {e}")
        
        self.feature_files = valid_files
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        try:
            # Load feature
            feature = np.load(self.feature_files[idx])
            
            # Convert to tensor
            feature = torch.from_numpy(feature).float()
            
            # Make sure feature has correct format for LSTM: [sequence_length, feature_dim]
            if len(feature.shape) == 1:  # Single feature vector
                feature = feature.unsqueeze(0)  # Add sequence dimension [feature_dim] -> [1, feature_dim]
            
            # Get label
            label = self.labels[idx]
            
            return feature, label
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading feature {self.feature_files[idx]}: {e}")
            
            # Return empty tensor and label as fallback
            return torch.zeros((1, self.input_size), dtype=torch.float32), self.labels[idx]

def parse_args():
    parser = argparse.ArgumentParser(description='Foundation Model + LSTM (Phase 2)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--features_dir', type=str, default='features/foundation',
                      help='Directory to save/load extracted features')
    parser.add_argument('--log_dir', type=str, default='logs/foundation_phase2',
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='models/foundation_phase2',
                      help='Directory to save models')
    
    # Feature extraction arguments
    parser.add_argument('--foundation_checkpoint', type=str, required=True,
                      help='Path to foundation model checkpoint')
    parser.add_argument('--extract_features', action='store_true',
                      help='Extract features before training')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for feature extraction')
    parser.add_argument('--num_frames', type=int, default=8,
                      help='Number of frames to sample per video')
    
    # LSTM arguments
    parser.add_argument('--input_size', type=int, default=768,
                      help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=512,
                      help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--patience', type=int, default=7,
                      help='Early stopping patience')
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                      choices=['cross_entropy', 'focal', 'weighted_cross_entropy'],
                      help='Type of loss function to use')
    parser.add_argument('--focal_alpha', type=float, default=0.55,
                      help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                      help='Gamma parameter for Focal Loss')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    
    # Evaluation arguments
    parser.add_argument('--skip_train', action='store_true',
                      help='Skip training and only evaluate')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to LSTM model checkpoint to load')
    
    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true',
                      help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='endofm-phase2',
                      help='WandB project name')
    
    # Config file for foundation model
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                      default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
    
    return parser.parse_args()

def extract_features(args, logger):
    """Extract features using the foundation model"""
    
    logger.info("Starting feature extraction process...")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Ensure deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Create feature directories
    feature_dir = Path(args.features_dir)
    train_dir = feature_dir / 'train'
    val_dir = feature_dir / 'val'
    test_dir = feature_dir / 'test'  # Also extract test features
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args)
    config.DATA.TRAIN_SAMPLING_METHOD = args.sampling_method
    config.DATA.VAL_SAMPLING_METHOD = args.sampling_method
    config.DATA.TEST_SAMPLING_METHOD = args.sampling_method
    config.DATA.NUM_FRAMES = args.num_frames
    
    # Create datasets and dataloaders
    logger.info(f"Creating datasets with {args.sampling_method} sampling method and {args.num_frames} frames")
    try:
        train_dataset = UCF101(cfg=config, mode="train", num_retries=10)
        logger.info(f"Created train dataset with {len(train_dataset)} samples")
        
        val_dataset = UCF101(cfg=config, mode="val", num_retries=10)
        logger.info(f"Created val dataset with {len(val_dataset)} samples")
        
        # Try to create test dataset (fallback to val if not available)
        try:
            test_dataset = UCF101(cfg=config, mode="test", num_retries=10)
            logger.info(f"Created test dataset with {len(test_dataset)} samples")
        except Exception as e:
            logger.info(f"No separate test dataset found: {str(e)}")
            logger.info("Using validation dataset for test")
            test_dataset = val_dataset
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False  # Keep order for feature extraction
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )
        
        logger.info(f"Created dataloaders with {len(train_loader)} train batches, {len(val_loader)} val batches, and {len(test_loader)} test batches")
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    
    # Create foundation model
    logger.info(f"Loading foundation model from {args.foundation_checkpoint}")
    try:
        # Create model
        model = get_vit_base_patch16_224(cfg=config, no_head=True)
        logger.info("Created foundation model")
        
        # Load pretrained weights
        if args.foundation_checkpoint.endswith('.pth') or args.foundation_checkpoint.endswith('.pt'):
            # Load checkpoint file
            checkpoint = torch.load(args.foundation_checkpoint, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "teacher" in checkpoint:
                    checkpoint = checkpoint["teacher"]
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                    # Handle DDP model
                    if all(k.startswith('module.') for k in checkpoint.keys()):
                        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            
            # Extract backbone weights
            if any(k.startswith('backbone.') for k in checkpoint.keys()):
                checkpoint = {k[len('backbone.'):]: v for k, v in checkpoint.items() 
                             if k.startswith('backbone.')}
            
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded foundation model with message: {msg}")
        else:
            logger.error(f"Unsupported checkpoint format: {args.foundation_checkpoint}")
            raise ValueError(f"Unsupported checkpoint format: {args.foundation_checkpoint}")
        
        # Move model to device
        model = model.cuda()
        model.eval()
        
        logger.info("Foundation model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading foundation model: {str(e)}")
        raise
    
    # Extract features for training set
    logger.info("Extracting features for training set...")
    extract_split_features(model, train_loader, train_dir, logger, args.num_frames)
    
    # Extract features for validation set
    logger.info("Extracting features for validation set...")
    extract_split_features(model, val_loader, val_dir, logger, args.num_frames)
    
    # Extract features for test set
    logger.info("Extracting features for test set...")
    extract_split_features(model, test_loader, test_dir, logger, args.num_frames)
    
    logger.info("Feature extraction completed")

def extract_split_features(model, dataloader, output_dir, logger, num_frames=8):
    """
    Extract features for a data split, ensuring proper shape for LSTM
    
    This improved version tracks features by video ID and organizes them into sequences
    """
    # Dictionary to collect features by video
    video_features = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices, meta) in enumerate(tqdm(dataloader, desc="Extracting features")):
            try:
                # Move inputs to GPU
                inputs = inputs.cuda(non_blocking=True)
                
                # Get foundation model features - shape will be [batch_size, feature_dim]
                features = model(inputs)
                
                # Debug the feature shape
                if batch_idx == 0:
                    logger.info(f"Foundation model output shape: {features.shape}")
                
                # Process each sample in the batch
                for i, idx in enumerate(indices):
                    video_id = idx.item()
                    label = targets[i].item()
                    
                    # Create structured features
                    feature_tensor = features[i].cpu().numpy()  # Single feature vector
                    
                    # Save the feature with structured filename: {video_id}_{label}.npy
                    np.save(output_dir / f"{video_id}_{label}.npy", feature_tensor)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx * dataloader.batch_size} samples")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    # Count extracted features
    feature_count = len(list(output_dir.glob('*.npy')))
    logger.info(f"Extracted {feature_count} features")

def train_lstm(args, logger):
    """Train LSTM on extracted features"""
    
    logger.info("Starting LSTM training...")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Ensure deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Generate unique model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.model_dir, f"lstm_model_{timestamp}.pth")
    logger.info(f"Model will be saved to: {model_save_path}")
    
    # Initialize WandB
    if args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                config=vars(args)
            )
            logger.info("Initialized WandB logging")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {str(e)}")
    
    # Create feature datasets
    train_dataset = FeatureDataset(args.features_dir, mode='train', logger=logger, input_size=args.input_size)
    val_dataset = FeatureDataset(args.features_dir, mode='val', logger=logger, input_size=args.input_size)
    
    # Create or use test dataset (using validation split if no separate test features exist)
    test_dir = Path(args.features_dir) / 'test'
    if test_dir.exists() and len(list(test_dir.glob('*.npy'))) > 0:
        test_dataset = FeatureDataset(args.features_dir, mode='test', logger=logger, input_size=args.input_size)
        logger.info(f"Using separate test dataset with {len(test_dataset)} samples")
    else:
        test_dataset = val_dataset
        logger.info("Using validation dataset as test dataset")
    
    # Check if we have valid datasets
    if len(train_dataset) == 0:
        logger.error("No training samples found. Make sure feature extraction worked correctly.")
        return None
    
    if len(val_dataset) == 0:
        logger.error("No validation samples found. Make sure feature extraction worked correctly.")
        return None
    
    # Log a sample feature shape to help debugging
    if len(train_dataset) > 0:
        sample_feature, _ = train_dataset[0]
        logger.info(f"Sample feature shape: {sample_feature.shape}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders with {len(train_loader)} train batches and {len(val_loader)} val batches")
    
    # Create LSTM model
    logger.info(f"Creating LSTM model with input_size={args.input_size}, hidden_size={args.hidden_size}")
    model = FoundationLSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_classes=args.num_classes
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        logger.info(f"Loading LSTM checkpoint from {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    # Setup loss function
    if args.loss_type == 'focal':
        logger.info(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        logger.info("Using Cross Entropy Loss")
        criterion = torch.nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training variables
    best_val_f1 = 0.0
    best_val_auprc = 0.0  # Also track AUPRC
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'val_auroc': [],
        'val_auprc': [],  # Add AUPRC tracking
        'val_confusion_matrix': []
    }
    
    # Train model
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            try:
                # Debug information - log input shape
                if len(inputs.shape) != 3:
                    logger.info(f"Input shape before reshaping: {inputs.shape}")
                    
                    # Handle individual feature vectors by adding sequence dimension
                    if len(inputs.shape) == 2:
                        if inputs.shape[0] == args.batch_size:
                            # Shape is [batch_size, feature_dim]
                            inputs = inputs.unsqueeze(1)  # Reshape to [batch_size, 1, feature_dim]
                            logger.info(f"Reshaped inputs to: {inputs.shape}")
                
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)
                
                # Store predictions and labels for metrics
                train_all_preds.extend(predicted.cpu().numpy())
                train_all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logger.error(f"Error in training batch: {str(e)}")
                continue
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total if train_total > 0 else float('inf')
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Calculate F1 score if predictions are available
        train_f1 = 0.0
        if train_all_preds and train_all_labels:
            try:
                train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
            except Exception as e:
                logger.error(f"Error calculating train F1: {str(e)}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        val_all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                try:
                    # Handle individual feature vectors
                    if len(inputs.shape) == 2:
                        if inputs.shape[0] == args.batch_size:
                            # Shape is [batch_size, feature_dim]
                            inputs = inputs.unsqueeze(1)  # Reshape to [batch_size, 1, feature_dim]
                    
                    # Move data to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Update statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
                    
                    # Store predictions and labels for metrics
                    val_all_preds.extend(predicted.cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())
                    
                    # For binary classification, store probabilities
                    if args.num_classes == 2:
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        val_all_probs.extend(probs.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue
        
        # Calculate validation metrics
        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Calculate F1 score and confusion matrix if predictions are available
        val_f1 = 0.0
        val_cm = np.zeros((args.num_classes, args.num_classes))
        
        if val_all_preds and val_all_labels:
            try:
                val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')
                val_cm = confusion_matrix(val_all_labels, val_all_preds, labels=range(args.num_classes))
            except Exception as e:
                logger.error(f"Error calculating validation metrics: {str(e)}")
        
        # Calculate additional metrics for binary classification
        val_auroc = 0.5  # Default value
        val_specificity = 0.0  # Default value
        val_sensitivity = 0.0  # Default value
        val_auprc = 0.0  # Default value
        
        if args.num_classes == 2 and len(val_all_probs) > 0 and len(np.unique(val_all_labels)) > 1:
            try:
                val_auroc = roc_auc_score(val_all_labels, val_all_probs)
                val_auprc = average_precision_score(val_all_labels, val_all_probs)
                
                # Calculate specificity and sensitivity from confusion matrix
                if val_cm.shape == (2, 2):
                    tn, fp, fn, tp = val_cm.ravel()
                    val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    val_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            except Exception as e:
                logger.error(f"Error calculating binary metrics: {str(e)}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['val_auroc'].append(val_auroc)
        history['val_auprc'].append(val_auprc)  # Track AUPRC
        history['val_confusion_matrix'].append(val_cm.tolist())
        
        # Log epoch results
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if args.num_classes == 2:
            logger.info(f"Val AUROC: {val_auroc:.4f}, Val Specificity: {val_specificity:.4f}, "
                        f"Val Sensitivity: {val_sensitivity:.4f}, Val AUPRC: {val_auprc:.4f}")  # Log AUPRC
        
        # Log to WandB
        if args.use_wandb:
            wandb_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if args.num_classes == 2:
                wandb_metrics.update({
                    'val_auroc': val_auroc,
                    'val_specificity': val_specificity,
                    'val_sensitivity': val_sensitivity,
                    'val_auprc': val_auprc  # Add AUPRC to WandB
                })
                
                # Add confusion matrix to WandB
                try:
                    wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=val_all_labels,
                            preds=val_all_preds,
                            class_names=[f"Class {i}" for i in range(args.num_classes)]
                        )
                    })
                except Exception as e:
                    logger.error(f"Error logging confusion matrix to WandB: {str(e)}")
            
            wandb.log(wandb_metrics)
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Early stopping check - using both F1 and AUPRC
        improvement = False
        if val_f1 > best_val_f1 or (val_auprc > best_val_auprc and args.num_classes == 2):
            if val_f1 > best_val_f1:
                logger.info(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}")
                best_val_f1 = val_f1
                improvement = True
            
            if val_auprc > best_val_auprc and args.num_classes == 2:
                logger.info(f"Validation AUPRC improved from {best_val_auprc:.4f} to {val_auprc:.4f}")
                best_val_auprc = val_auprc
                improvement = True
            
            if improvement:
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_auroc': val_auroc,
                    'val_auprc': val_auprc,  # Save AUPRC
                    'history': history,
                    'args': vars(args)
                }, model_save_path)
                logger.info(f"Saved best model to {model_save_path}")
                
                # Plot and save confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=range(args.num_classes), 
                            yticklabels=range(args.num_classes))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.savefig(os.path.join(args.log_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
                plt.close()
        else:
            patience_counter += 1
            logger.info(f"Validation F1 did not improve. Patience: {patience_counter}/{args.patience}")
    
    # Training completed
    logger.info("LSTM training completed!")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'training_history.png'))
    plt.close()
    
    # Save history to file
    with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
        import json
        json.dump(history, f)
    
    # Test the best model on the test set
    logger.info("Loading best model for evaluation on test set...")
    try:
        best_checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        
        # Create a test dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        # Evaluate on test set
        test_metrics = evaluate_lstm_on_test_set(model, test_loader, criterion, args, logger, device)
        
        # Final summary
        logger.info("LSTM Model (Phase 2) training completed")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
        logger.info(f"Test PR-AUC: {test_metrics['auprc']:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating model on test set: {str(e)}")
    
    # Close WandB
    if args.use_wandb:
        wandb.finish()
    
    # Return best model path
    return model_save_path

def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, f'phase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('foundation_phase2')
    logger.info("Starting Phase 2: Foundation Model + LSTM")
    logger.info(f"Arguments: {vars(args)}")
    
    # Step 1: Extract features if requested
    if args.extract_features:
        logger.info("Starting feature extraction...")
        extract_features(args, logger)
    else:
        logger.info("Skipping feature extraction...")
    
    # Step 2: Train LSTM on extracted features
    if not args.skip_train:
        logger.info("Starting LSTM training...")
        lstm_model_path = train_lstm(args, logger)
        logger.info(f"LSTM training completed. Best model saved to {lstm_model_path}")
    else:
        logger.info("Skipping LSTM training...")
    
    logger.info("Phase 2 completed!")

if __name__ == "__main__":
    main()