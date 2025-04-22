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

from models import get_vit_base_patch16_224
from datasets import UCF101
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from utils.focal_loss import FocalLoss
from utils.custom_sampling import set_all_random_seeds
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

def evaluate_on_test_set(model, test_loader, criterion, args, logger):
    """Evaluate model on test set with comprehensive metrics"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_all_preds = []
    test_all_labels = []
    test_all_probs = []
    
    logger.info("Evaluating model on test set...")
    with torch.no_grad():
        for inputs, targets, _, _ in test_loader:
            # Move data to GPU
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(targets).sum().item()
            test_total += targets.size(0)
            
            # Store predictions and labels for metrics
            test_all_preds.extend(predicted.cpu().numpy())
            test_all_labels.extend(targets.cpu().numpy())
            
            # For binary classification, store probabilities
            if args.num_classes == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                test_all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total
    
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
    plt.savefig(os.path.join(args.log_dir, 'test_confusion_matrix.png'))
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
        plt.savefig(os.path.join(args.log_dir, 'test_roc_curve.png'))
        plt.close()
        
        # PR Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(test_all_labels, test_all_probs)
        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {test_auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Set Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(args.log_dir, 'test_pr_curve.png'))
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

def parse_args():
    parser = argparse.ArgumentParser(description='Foundation Model Fine-tuning on Frames (Phase 1)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='logs/foundation_phase1',
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='models/foundation_phase1',
                      help='Directory to save models')
    
    # Training arguments
    parser.add_argument('--train_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for testing')
    parser.add_argument('--num_frames', type=int, default=8,
                      help='Number of frames to sample per video')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--patience', type=int, default=7,
                      help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='vit_base',
                      help='Architecture')
    parser.add_argument('--patch_size', type=int, default=16,
                      help='Patch size')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                      help='Path to pretrained weights')
    parser.add_argument('--scratch', action='store_true',
                      help='Enable frame adaptation for pretrained weights with different frame counts')
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                      choices=['cross_entropy', 'focal', 'weighted_cross_entropy'],
                      help='Type of loss function to use')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                      help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Gamma parameter for Focal Loss')
    
    # Evaluation arguments
    parser.add_argument('--skip_train', action='store_true',
                      help='Skip training and only evaluate')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to model checkpoint to load')
    
    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true',
                      help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='endofm-phase1',
                      help='WandB project name')
    
    # Distributed training
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    
    # Config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                      default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_all_random_seeds(args.seed)
    
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    
    # Ensure deterministic behavior
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('foundation_phase1')
    logger.info("Starting Phase 1: Foundation Model fine-tuning on frames")
    logger.info(f"Arguments: {vars(args)}")
    
    # Generate unique model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.model_dir, f"foundation_model_{timestamp}.pth")
    logger.info(f"Model will be saved to: {model_save_path}")
    
    # Initialize WandB
    if args.use_wandb and utils.is_main_process():
        try:
            wandb.init(
                project=args.wandb_project,
                config=vars(args)
            )
            logger.info("Initialized WandB logging")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {str(e)}")
    
    # Load configuration
    config = load_config(args)
    config.DATA.TRAIN_SAMPLING_METHOD = args.train_sampling
    config.DATA.VAL_SAMPLING_METHOD = args.val_sampling
    config.DATA.TEST_SAMPLING_METHOD = args.test_sampling
    config.DATA.NUM_FRAMES = args.num_frames
    
    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    try:
        train_dataset = UCF101(cfg=config, mode="train", num_retries=10)
        logger.info(f"Created train dataset with {len(train_dataset)} samples")
        
        val_dataset = UCF101(cfg=config, mode="val", num_retries=10)
        logger.info(f"Created val dataset with {len(val_dataset)} samples")
        
        # Create test dataset (using validation split if no separate test split exists)
        try:
            test_dataset = UCF101(cfg=config, mode="test", num_retries=10)
            logger.info(f"Created test dataset with {len(test_dataset)} samples")
        except Exception as e:
            logger.info(f"No separate test dataset found: {str(e)}")
            logger.info("Using validation dataset as test dataset")
            test_dataset = val_dataset
        
        # Create distributed sampler for training
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
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
    
    # Create model
    logger.info(f"Creating foundation model ({args.arch})...")
    try:
        model = get_vit_base_patch16_224(cfg=config, no_head=False)
        
        # Load pretrained weights
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
        if "teacher" in checkpoint:
            checkpoint = checkpoint["teacher"]
        
        # Extract backbone weights
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in checkpoint.items() 
                             if x.startswith("backbone.")}
        
        # Check if --scratch flag is enabled or we're using different frame counts
        if args.scratch:
            logger.info("Scratch mode enabled - checking for time embedding adaptation needs")
            
            # Check for time_embed size mismatch and adapt if necessary
            if 'time_embed' in renamed_checkpoint:
                pretrained_time_embed = renamed_checkpoint['time_embed']
                current_time_frames = args.num_frames
                pretrained_time_frames = pretrained_time_embed.shape[1]
                
                if current_time_frames != pretrained_time_frames:
                    logger.info(f"Adapting time_embed from {pretrained_time_frames} frames to {current_time_frames} frames")
                    
                    # Common variables for both cases
                    channels = pretrained_time_embed.shape[2]
                    
                    # Handle time embedding mismatch
                    if current_time_frames > pretrained_time_frames:
                        # Expand by repeating the pattern and interpolating (upsampling)
                        expanded_time_embed = torch.zeros((1, current_time_frames, channels))
                        
                        # Linear interpolation of time embeddings to new size
                        for c in range(channels):
                            # Extract 1D signal for this channel
                            signal = pretrained_time_embed[0, :, c].numpy()
                            # Create interpolator
                            from scipy import interpolate
                            x_original = np.linspace(0, 1, pretrained_time_frames)
                            x_new = np.linspace(0, 1, current_time_frames)
                            f = interpolate.interp1d(x_original, signal, kind='linear')
                            # Interpolate to get new signal
                            new_signal = f(x_new)
                            # Store in expanded tensor
                            expanded_time_embed[0, :, c] = torch.tensor(new_signal)
                        
                        # Replace with expanded version
                        renamed_checkpoint['time_embed'] = expanded_time_embed
                        logger.info(f"Expanded time_embed to {expanded_time_embed.shape}")
                    else:
                        # Downsample by interpolation (reducing frames)
                        contracted_time_embed = torch.zeros((1, current_time_frames, channels))
                        
                        # Linear interpolation to fewer frames
                        for c in range(channels):
                            # Extract 1D signal for this channel
                            signal = pretrained_time_embed[0, :, c].numpy()
                            # Create interpolator
                            from scipy import interpolate
                            x_original = np.linspace(0, 1, pretrained_time_frames)
                            x_new = np.linspace(0, 1, current_time_frames)
                            f = interpolate.interp1d(x_original, signal, kind='linear')
                            # Interpolate to get new signal
                            new_signal = f(x_new)
                            # Store in contracted tensor
                            contracted_time_embed[0, :, c] = torch.tensor(new_signal)
                        
                        # Replace with contracted version
                        renamed_checkpoint['time_embed'] = contracted_time_embed
                        logger.info(f"Contracted time_embed from {pretrained_time_embed.shape} to {contracted_time_embed.shape}")
        
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        logger.info(f"Loaded pretrained weights with message: {msg}")
        
        # Modify the head for our number of classes
        if hasattr(model, 'head'):
            model.head = torch.nn.Linear(model.embed_dim, args.num_classes)
            logger.info(f"Modified head for {args.num_classes} classes")
        
        # Move model to device
        model = model.cuda()
        
        # Use DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
        logger.info(f"Model created and initialized")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
    
    # Setup loss function
    if args.loss_type == 'focal':
        logger.info(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_type == 'weighted_cross_entropy':
        # Calculate class weights if needed
        logger.info("Using Weighted Cross Entropy Loss")
        criterion = torch.nn.CrossEntropyLoss(weight=None)
    else:
        logger.info("Using Cross Entropy Loss")
        criterion = torch.nn.CrossEntropyLoss()
    
    # Setup optimizer (with lower learning rate for pretrained parts)
    param_groups = [
        {'params': model.module.blocks.parameters(), 'lr': args.learning_rate * 0.1},  # Lower LR for pretrained parts
        {'params': model.module.head.parameters(), 'lr': args.learning_rate}  # Higher LR for new head
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_auprc = 0.0  # Track AUPRC for early stopping (new)
    patience_counter = 0
    
    # Initialize training history
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
    
    # Training loop
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Set epoch for sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        
        for i, (inputs, targets, _, _) in enumerate(train_loader):
            # Move data to GPU
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
            
            # Store predictions and labels for metrics
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(targets.cpu().numpy())
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        val_all_probs = []
        
        with torch.no_grad():
            for i, (inputs, targets, _, _) in enumerate(val_loader):
                # Move data to GPU
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
                
                # Store predictions and labels for metrics
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(targets.cpu().numpy())
                
                # For binary classification, store probabilities for ROC-AUC and AUPRC
                if args.num_classes == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    val_all_probs.extend(probs.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')
        val_cm = confusion_matrix(val_all_labels, val_all_preds)
        
        # Calculate additional metrics for binary classification
        val_auroc = 0.5  # Default value
        val_auprc = 0.0  # Default value
        val_specificity = 0.0  # Default value
        val_sensitivity = 0.0  # Default value
        
        if args.num_classes == 2 and len(np.unique(val_all_labels)) > 1:
            try:
                # Calculate AUROC
                val_auroc = roc_auc_score(val_all_labels, val_all_probs)
                
                # Calculate AUPRC explicitly
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
        history['val_auprc'].append(val_auprc)  # Add AUPRC
        history['val_confusion_matrix'].append(val_cm.tolist())
        
        # Log epoch results
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if args.num_classes == 2:
            logger.info(f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
        
        # Log to WandB
        if args.use_wandb and utils.is_main_process():
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
                    'val_auprc': val_auprc,  # Add AUPRC to WandB logs
                    'val_specificity': val_specificity,
                    'val_sensitivity': val_sensitivity
                })
                
                # Log confusion matrix to WandB
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
        scheduler.step()
        
        # Early stopping check (using both F1 and AUPRC)
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
                if utils.is_main_process():
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
                if utils.is_main_process():
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=range(args.num_classes),
                                yticklabels=range(args.num_classes))
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                    plt.savefig(os.path.join(args.log_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
                    plt.close()
                    
                    # Save normalized confusion matrix
                    if val_cm.sum() > 0:
                        plt.figure(figsize=(10, 8))
                        cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
                        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                                    xticklabels=range(args.num_classes),
                                    yticklabels=range(args.num_classes))
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Normalized Confusion Matrix - Epoch {epoch+1}')
                        plt.savefig(os.path.join(args.log_dir, f'norm_confusion_matrix_epoch_{epoch+1}.png'))
                        plt.close()
                        
                    # For binary classification, also create ROC and PR curves
                    if args.num_classes == 2 and len(val_all_probs) > 0:
                        # ROC Curve
                        plt.figure(figsize=(10, 8))
                        from sklearn.metrics import roc_curve
                        fpr, tpr, _ = roc_curve(val_all_labels, val_all_probs)
                        plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {val_auroc:.4f})')
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve - Epoch {epoch+1}')
                        plt.legend()
                        plt.savefig(os.path.join(args.log_dir, f'roc_curve_epoch_{epoch+1}.png'))
                        plt.close()
                        
                        # PR Curve
                        plt.figure(figsize=(10, 8))
                        precision, recall, _ = precision_recall_curve(val_all_labels, val_all_probs)
                        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {val_auprc:.4f})')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title(f'Precision-Recall Curve - Epoch {epoch+1}')
                        plt.legend()
                        plt.savefig(os.path.join(args.log_dir, f'pr_curve_epoch_{epoch+1}.png'))
                        plt.close()
        else:
            patience_counter += 1
            logger.info(f"Validation metrics did not improve. Patience: {patience_counter}/{args.patience}")
    
    # Training completed
    logger.info("Training completed!")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Best validation AUPRC: {best_val_auprc:.4f}")
    
    # Plot training history
    if utils.is_main_process():
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(2, 3, 3)
        plt.plot(history['train_f1'], label='Train')
        plt.plot(history['val_f1'], label='Validation')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.legend()
        
        if args.num_classes == 2:
            plt.subplot(2, 3, 4)
            plt.plot(history['val_auroc'], label='AUROC')
            plt.title('ROC AUC')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(2, 3, 5)
            plt.plot(history['val_auprc'], label='AUPRC')
            plt.title('PR AUC')
            plt.xlabel('Epoch')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.log_dir, 'training_history.png'))
        plt.close()
        
        # Save history to file
        with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
            import json
            json.dump(history, f)
    
    # Close WandB
    if args.use_wandb and utils.is_main_process():
        wandb.finish()
    
    # Test the best model on the test set
    logger.info("Loading best model for evaluation on test set...")
    if utils.is_main_process():
        try:
            best_checkpoint = torch.load(model_save_path, map_location='cpu')
            model.load_state_dict(best_checkpoint["model_state_dict"])
            
            # Evaluate on test set
            test_metrics = evaluate_on_test_set(model, test_loader, criterion, args, logger)
            
            # Final summary
            logger.info("Foundation Model (Phase 1) training completed")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
            logger.info(f"Test PR-AUC: {test_metrics['auprc']:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating model on test set: {str(e)}")
    
    # Return best model path for Phase 2
    return model_save_path

if __name__ == "__main__":
    main()