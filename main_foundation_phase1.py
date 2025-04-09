# main_foundation_phase1.py

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

from models import get_vit_base_patch16_224
from datasets import UCF101
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from utils.focal_loss import FocalLoss
from utils.custom_sampling import set_all_random_seeds
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score

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
        
        logger.info(f"Created dataloaders with {len(train_loader)} train batches and {len(val_loader)} val batches")
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
        
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        logger.info(f"Loaded pretrained weights with message: {msg}")
        
        # Modify the head for our number of classes
        if hasattr(model, 'head'):
            model.head = torch.nn.Linear(model.embed_dim, args.num_classes)
            logger.info(f"Modified head for {args.num_classes} classes")
        
        # Move model to device
        model = model.cuda()
        
        # Use DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        
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
    patience_counter = 0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
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
                
                # For binary classification, store probabilities for ROC-AUC
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
        
        if args.num_classes == 2 and len(np.unique(val_all_labels)) > 1:
            try:
                val_auroc = roc_auc_score(val_all_labels, val_all_probs)
                val_auprc = average_precision_score(val_all_labels, val_all_probs)
            except:
                pass
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
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
                    'val_auprc': val_auprc
                })
            
            wandb.log(wandb_metrics)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping check (using F1 score)
        if val_f1 > best_val_f1:
            logger.info(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}")
            best_val_f1 = val_f1
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
                    'history': history,
                    'args': vars(args)
                }, model_save_path)
                logger.info(f"Saved best model to {model_save_path}")
        else:
            patience_counter += 1
            logger.info(f"Validation F1 did not improve. Patience: {patience_counter}/{args.patience}")
    
    # Training completed
    logger.info("Training completed!")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    
    # Close WandB
    if args.use_wandb and utils.is_main_process():
        wandb.finish()
    
    # Return best model path for Phase 2
    return model_save_path

if __name__ == "__main__":
    main()