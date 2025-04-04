import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, average_precision_score

from datasets import UCF101, HMDB51, Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from utils.focal_loss import FocalLoss  # Import the FocalLoss class
from utils.custom_sampling import set_all_random_seeds, FrameSampler  # Import sampling utilities
def save_all_sampling_indices(datasets, output_dir):
    """Save sampling indices for all datasets."""
    for split, dataset in datasets.items():
        if hasattr(dataset, 'save_sampling_indices'):
            try:
                dataset.save_sampling_indices()
                print(f"Saved sampling indices for {split} dataset")
            except Exception as e:
                print(f"Error saving sampling indices for {split} dataset: {str(e)}")

def eval_finetune(args):
    # Set seed for complete reproducibility
    set_all_random_seeds(42)
    print("Set all random seeds to 42 for reproducibility")
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = False  # Ensure deterministic behavior
    cudnn.deterministic = True  # Ensure deterministic behavior
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # Create directory for sampling indices CSVs
    sampling_csv_dir = os.path.join(args.output_dir, 'sampling_indices')
    os.makedirs(sampling_csv_dir, exist_ok=True)
    
    # Configure logger
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('endo_fm')
    logger.info("Starting training with seed 42 for reproducible sampling")

    # ============ preparing data ... ============
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    
    # Add sampling method to config
    config.DATA.TRAIN_SAMPLING_METHOD = args.train_sampling
    config.DATA.VAL_SAMPLING_METHOD = args.val_sampling  
    config.DATA.TEST_SAMPLING_METHOD = args.test_sampling
    config.DATA.NUM_FRAMES = args.num_frames
    config.DATA.SEED = 42  # Fixed seed for reproducibility
    config.DATA.CSV_SAVE_DIR = sampling_csv_dir
    global_logger = logger
    
    datasets = {}
    
    if args.dataset == "ucf101":
        # Initialize with sampling tracker
        dataset_train = UCF101(cfg=config, mode="train", num_retries=10)
        if hasattr(dataset_train, 'init_sampler'):
            dataset_train.init_sampler(sampling_csv_dir, global_logger, "ucf101", "train", args.train_sampling)
        datasets['train'] = dataset_train
            
        dataset_val = UCF101(cfg=config, mode="val", num_retries=10)
        if hasattr(dataset_val, 'init_sampler'):
            dataset_val.init_sampler(sampling_csv_dir, global_logger, "ucf101", "val", args.val_sampling)
        datasets['val'] = dataset_val
            
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        if hasattr(dataset_train, 'init_sampler'):
            dataset_train.init_sampler(sampling_csv_dir, global_logger, "hmdb51", "train", args.train_sampling)
        datasets['train'] = dataset_train
            
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        if hasattr(dataset_val, 'init_sampler'):
            dataset_val.init_sampler(sampling_csv_dir, global_logger, "hmdb51", "val", args.val_sampling)
        datasets['val'] = dataset_val
            
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        if hasattr(dataset_train, 'init_sampler'):
            dataset_train.init_sampler(sampling_csv_dir, global_logger, "kinetics400", "train", args.train_sampling)
        datasets['train'] = dataset_train
            
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
        if hasattr(dataset_val, 'init_sampler'):
            dataset_val.init_sampler(sampling_csv_dir, global_logger, "kinetics400", "val", args.val_sampling)
        datasets['val'] = dataset_val
            
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    print(f"Using sampling methods: Train={args.train_sampling}, Val={args.val_sampling}, Test={args.test_sampling}")

    # Save sampling indices after initial data loading
    save_all_sampling_indices(datasets, args.output_dir)

    # ============ building network ... ============
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        else:
            raise Exception(f"invalid model: {args.arch}")

    if not args.scratch and args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")
    elif args.scratch:
        # Load the checkpoint for inspection before applying it to the model
        ckpt = torch.load(args.pretrained_weights, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]

        # Create renamed checkpoint
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}

        # Check for time_embed size mismatch and adapt if necessary
        if 'time_embed' in renamed_checkpoint:
            pretrained_time_embed = renamed_checkpoint['time_embed']
            current_time_frames = args.num_frames
            pretrained_time_frames = pretrained_time_embed.shape[1]
            
            if current_time_frames != pretrained_time_frames:
                print(f"Adapting time_embed from {pretrained_time_frames} frames to {current_time_frames} frames")
                
                # Handle time embedding mismatch
                if current_time_frames > pretrained_time_frames:
                    # Expand by repeating the pattern and interpolating
                    channels = pretrained_time_embed.shape[2]
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
                    print(f"Expanded time_embed to {expanded_time_embed.shape}")

        # Now load the model with the potentially modified checkpoint
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")

    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate

    linear_classifier = LinearClassifier(model_embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                         num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    if args.test:
        utils.restart_from_checkpoint(
            args.pretrained_model_weights,
            backbone_state_dict=model,
            state_dict=linear_classifier,
        )
        test_stats, metrics = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                          args.avgpool_patchtokens)
        test_stats, metrics = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                  args.avgpool_patchtokens)
        
        print(f"Metrics of the network on the {len(dataset_val)} test images:")
        print(f"Specificity: {metrics['specificity'] * 100:.1f}%")
        print(f"Sensitivity: {metrics['sensitivity'] * 100:.1f}%")
        print(f"F1 score: {metrics['f1'] * 100:.1f}%")
        print(f"AUROC: {metrics['auroc'] * 100:.1f}%")
        print(f"AUPRC: {metrics['auprc'] * 100:.1f}%")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

        # Calculate the combined score
        # Case 1: Check for zero specificity (penalize heavily)
        if metrics['specificity'] == 0:
            # Use a strong penalty but still consider other metrics at lower weights
            combined_score = (0.1 * metrics['f1']) + (0.05 * metrics['auroc']) + (0.05 * metrics['auprc'])
            logger.info(f"Applied zero-specificity penalty, reducing combined score to {combined_score:.4f}")
        else:
            # Prioritize specificity even more when it exists
            combined_score = (0.7 * metrics['specificity']) + (0.15 * metrics['f1']) + (0.1 * metrics['auroc']) + (0.05 * metrics['auprc'])
        print(f"Combined score: {combined_score:.4f}")

        # For binary classification, provide detailed breakdown
        if args.num_labels == 2:
            tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
            print(f"Binary classification details:")
            print(f"  True Negatives: {tn}, False Positives: {fp}")
            print(f"  False Negatives: {fn}, True Positives: {tp}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
        
        # Save final sampling indices
        save_all_sampling_indices(datasets, args.output_dir)
        
        exit(0)

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # Set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'lr': scaled_lr},
         {'params': linear_classifier.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Setup loss function based on arguments (cross-entropy or focal loss)
    if args.loss_function == 'focal_loss':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    # Optionally resume from a checkpoint
    to_restore = {
        "epoch": 0, 
        "best_acc": 0., 
        "best_f1": 0., 
        "best_auroc": 0., 
        "best_auprc": 0.,
        "best_specificity": 0.,
        "best_sensitivity": 0.,
        "best_combined_score": 0.
    }
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_f1 = to_restore.get("best_f1", 0.0)
    best_auroc = to_restore.get("best_auroc", 0.0)
    best_auprc = to_restore.get("best_auprc", 0.0)
    best_specificity = to_restore.get("best_specificity", 0.0)
    best_sensitivity = to_restore.get("best_sensitivity", 0.0)
    best_combined_score = to_restore.get("best_combined_score", 0.0)
    best_epoch = start_epoch - 1  # Track the epoch with the best performance

    exploration_frequency = 5  # Try a fresh exploration every 5 epochs
    best_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
    logger.info(f"Will reload best checkpoint every {exploration_frequency} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        # Determine if this should be an exploration epoch
        is_exploration_epoch = (epoch % exploration_frequency == 0)
        
        # Reload the best model checkpoint if it exists (after epoch 0) and not in exploration mode
        if epoch > 0 and os.path.exists(best_checkpoint_path) and not is_exploration_epoch:
            logger.info(f"Reloading best model before starting epoch {epoch}")
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            
            # Load model and classifier weights
            model.load_state_dict(checkpoint["backbone_state_dict"])
            linear_classifier.load_state_dict(checkpoint["state_dict"])
            
            # Don't reset optimizer and scheduler to maintain learning rate progression
            
            logger.info(f"Continuing training from best model (specificity: {checkpoint.get('best_specificity', 0):.4f}, F1: {checkpoint.get('best_f1', 0):.4f})")
        elif is_exploration_epoch:
            logger.info(f"Exploration epoch {epoch}: Continuing with current model state without reloading best checkpoint")
        
        # Set the epoch for the data sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train(args, model, linear_classifier, criterion, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        
        # Update learning rate
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, metrics = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Epoch {epoch} evaluation metrics:")
            print(f"F1 score: {metrics['f1'] * 100:.1f}%")
            print(f"AUROC: {metrics['auroc'] * 100:.1f}%")
            print(f"AUPRC: {metrics['auprc'] * 100:.1f}%")
            
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'metric_{k}': v for k, v in metrics.items() if k != 'confusion_matrix'}}

            # Define a combined score that prioritizes specificity while not compromising other metrics too much
            # TODO:Calculate combined score giving higher weight to specificity (adjust weights as needed)
            combined_score = (0.35 * metrics['specificity']) + (0.3 * metrics['sensitivity']) + (0.2 * metrics['f1']) + (0.1 * metrics['auroc']) + (0.05 * metrics['auprc'])

            # For binary classification, give bonus to models that have at least some true negatives
            if args.num_labels == 2:
                tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
                
                # Give a bonus to the score if we have true negatives
                if tn > 0:
                    previous_score = combined_score
                    combined_score *= (1.0 + 0.1 * min(tn, 5) / 5.0)  # Up to 10% bonus for having true negatives
                    logger.info(f"Applied true negative bonus: {previous_score:.4f} → {combined_score:.4f}")
                    
                # Log confusion matrix statistics for better tracking
                logger.info(f"CM stats: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            # Log the combined score
            logger.info(f"Epoch {epoch} - Combined score: {combined_score:.4f} (spec: {metrics['specificity']:.4f}, f1: {metrics['f1']:.4f})")

            # Store confusion matrix as a string for logging
            cm_str = np.array2string(metrics['confusion_matrix'], 
                                    separator=', ', 
                                    prefix='', 
                                    suffix='', 
                                    threshold=np.inf, 
                                    edgeitems=np.inf,
                                    formatter={'int': lambda x: f"{x:4d}"})

            # Log confusion matrix with clearer formatting
            logger.info(f"Confusion Matrix (epoch {epoch}):\n{cm_str}")

            # Calculate and log specificity for each class from confusion matrix
            if args.num_labels == 2:
                tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
                logger.info(f"Binary classification details:")
                logger.info(f"  True Negatives: {tn}, False Positives: {fp}")
                logger.info(f"  False Negatives: {fn}, True Positives: {tp}")
                logger.info(f"  Specificity (TNR): {metrics['specificity']:.4f}, Sensitivity (TPR): {metrics['sensitivity']:.4f}")

            # Define minimum thresholds for improvement
            MIN_SCORE_IMPROVEMENT = 0.01  # 1% improvement
            MIN_SPECIFICITY_IMPROVEMENT = 0.05  # 5% improvement

            # Check if this is an improvement that meets our thresholds
            specificity_improved = metrics['specificity'] >= (best_specificity + MIN_SPECIFICITY_IMPROVEMENT)
            score_improved = combined_score >= (best_combined_score + MIN_SCORE_IMPROVEMENT)

            # The baseline minimum requirement: high specificity or significant combined score improvement
            save_condition = False

            # Case 1: Significant improvement in specificity (prioritize this)
            if specificity_improved:
                save_reason = f"specificity improved significantly: {metrics['specificity']:.4f} vs previous {best_specificity:.4f}"
                save_condition = True
            # NEW CASE: Significant improvement in sensitivity with acceptable specificity
            elif metrics['sensitivity'] > (best_sensitivity * 1.2) and metrics['specificity'] >= 0.6:
                save_reason = f"significant sensitivity improvement: {metrics['sensitivity']:.4f} vs {best_sensitivity:.4f} with acceptable specificity"
                save_condition = True
            # Case 2: Specificity stable but combined score improved significantly
            elif metrics['specificity'] >= best_specificity * 0.95 and score_improved:
                save_reason = f"combined score improved significantly while maintaining specificity"
                save_condition = True
            # Case 3: First model or major score improvement (>5%)
            elif best_combined_score == 0 or combined_score >= best_combined_score * 1.05:
                save_reason = f"major improvement in combined score: {combined_score:.4f} vs previous {best_combined_score:.4f}"
                save_condition = True
            # Case 4: Similar specificity but much better other metrics
            elif metrics['specificity'] >= best_specificity * 0.9 and metrics['f1'] >= best_f1 * 1.1:
                save_reason = f"much better F1 ({metrics['f1']:.4f} vs {best_f1:.4f}) with acceptable specificity"
                save_condition = True

            if save_condition and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    f.write(f"Epoch {epoch} confusion matrix:\n{cm_str}\n")
                    f.write(f"Specificity: {metrics['specificity']:.4f}, Sensitivity: {metrics['sensitivity']:.4f}\n")
                    f.write(f"Saved model because: {save_reason}\n")
                
                logger.info(f"Saving new model because: {save_reason}")
                
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "backbone_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_combined_score": combined_score,
                    "best_specificity": metrics['specificity'],
                    "best_sensitivity": metrics['sensitivity'],
                    "best_f1": metrics['f1'],
                    "best_auroc": metrics['auroc'],
                    "best_auprc": metrics['auprc'],
                    "train_sampling": args.train_sampling,
                    "val_sampling": args.val_sampling,
                    "test_sampling": args.test_sampling,
                    "loss_function": args.loss_function,
                    "seed": 42,
                    "save_reason": save_reason,  # Record why we saved this model
                }
                if args.loss_function == 'focal_loss':
                    save_dict["focal_alpha"] = args.focal_alpha
                    save_dict["focal_gamma"] = args.focal_gamma
                
                # Save the model
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
                
                # Also save confusion matrix visualization for the best model
                plt.figure(figsize=(10, 8))
                sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title(f'Best Model Confusion Matrix (Epoch {epoch})\n{save_reason}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(args.output_dir, f'best_confusion_matrix_epoch_{epoch}.png'))
                
                # Update best metrics and epoch tracking
                best_epoch = epoch
                best_combined_score = combined_score
                best_specificity = metrics['specificity']
                best_sensitivity = metrics['sensitivity']
                best_f1 = metrics['f1']
                best_auroc = metrics['auroc']
                best_auprc = metrics['auprc']
                
                logger.info(f"New best model saved! Combined score: {combined_score:.4f}")

            # log current metrics versus best
            logger.info(f'Current metrics:')
            logger.info(f'  Specificity: {metrics["specificity"] * 100:.1f}% (best: {best_specificity * 100:.1f}%)')
            logger.info(f'  F1: {metrics["f1"] * 100:.1f}% (best: {best_f1 * 100:.1f}%)')
            logger.info(f'  AUROC: {metrics["auroc"] * 100:.1f}% (best: {best_auroc * 100:.1f}%)')
            logger.info(f'  AUPRC: {metrics["auprc"] * 100:.1f}% (best: {best_auprc * 100:.1f}%)')
                
            print(f'Best metrics so far:')
            print(f'F1: {best_f1 * 100:.1f}%')
            print(f'AUROC: {best_auroc * 100:.1f}%')
            print(f'AUPRC: {best_auprc * 100:.1f}%')
            
            # Save updated sampling indices after each validation
            save_all_sampling_indices(datasets, args.output_dir)

    # After all epochs, load the best model and generate final metrics and confusion matrix
    if not args.test and utils.is_main_process():
        logger.info("\nTraining complete. Loading best model for final evaluation...")
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth.tar"),
            backbone_state_dict=model,
            state_dict=linear_classifier,
        )
        
        final_stats, final_metrics = validate_network(val_loader, model, linear_classifier, 
                                                args.n_last_blocks, args.avgpool_patchtokens)
        
        # Calculate final combined score
        final_combined_score = (0.5 * final_metrics['specificity']) + (0.25 * final_metrics['f1']) + (0.15 * final_metrics['auroc']) + (0.10 * final_metrics['auprc'])
        
        logger.info(f"Final metrics with best model (from epoch {best_epoch}):")
        logger.info(f"Combined score: {final_combined_score:.4f}")
        logger.info(f"Specificity: {final_metrics['specificity'] * 100:.1f}%")
        logger.info(f"Sensitivity: {final_metrics['sensitivity'] * 100:.1f}%")
        logger.info(f"F1 score: {final_metrics['f1'] * 100:.1f}%")
        logger.info(f"AUROC: {final_metrics['auroc'] * 100:.1f}%")
        logger.info(f"AUPRC: {final_metrics['auprc'] * 100:.1f}%")
        
        # Store confusion matrix as a string for logging
        cm_str = np.array2string(final_metrics['confusion_matrix'], 
                                separator=', ', 
                                prefix='', 
                                suffix='', 
                                threshold=np.inf, 
                                edgeitems=np.inf,
                                formatter={'int': lambda x: f"{x:4d}"})
        
        logger.info(f"Final confusion matrix:\n{cm_str}")
        
        # For binary classification, provide additional detailed breakdown
        if args.num_labels == 2:
            tn, fp, fn, tp = final_metrics['confusion_matrix'].ravel()
            logger.info(f"Final binary classification details:")
            logger.info(f"  True Negatives: {tn}, False Positives: {fp}")
            logger.info(f"  False Negatives: {fn}, True Positives: {tp}")
        
        # Plot and save confusion matrix for best overall model
        plt.figure(figsize=(10, 8))
        sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Confusion Matrix (Best Model from Epoch {best_epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'final_confusion_matrix.png'))
        print(f"Final confusion matrix saved to {os.path.join(args.output_dir, 'final_confusion_matrix.png')}")
        
        # Save final sampling indices
        save_all_sampling_indices(datasets, args.output_dir)


def train(args, model, linear_classifier, criterion, optimizer, loader, epoch, n, avgpool):
    model.train()
    linear_classifier.train()
    
    # Using tqdm for better progress display
    total_batches = len(loader)
    progress_bar = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", 
                        total=total_batches, unit="batch")
    
    loss_meter = utils.SmoothedValue(window_size=10, fmt='{avg:.4f}')
    lr_meter = utils.SmoothedValue(window_size=10, fmt='{value:.6f}')
    
    for inp, target, sample_idx, meta in progress_bar:
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(inp)
        output = linear_classifier(output)

        # compute loss using the chosen criterion
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # Update meters
        torch.cuda.synchronize()
        loss_meter.update(loss.item())
        lr_meter.update(optimizer.param_groups[0]["lr"])
        
        # Update progress bar
        progress_bar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'loss': f"{loss_meter.avg:.4f}"
        })
    
    # Print summary stats at the end of the epoch
    print(f"Epoch [{epoch}/{args.epochs}] completed. Train loss: {loss_meter.avg:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return {"loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    model.eval()
    linear_classifier.eval()
    
    # Use tqdm for better progress display
    progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch")
    
    loss_meter = utils.SmoothedValue(window_size=10, fmt='{avg:.4f}')
    acc1_meter = utils.SmoothedValue(window_size=10, fmt='{avg:.2f}')
    
    all_targets = []
    all_outputs = []
    all_probs = []  # Store prediction probabilities for ROC and PR curves
    
    for inp, target, sample_idx, meta in progress_bar:
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        # Store targets and softmax probabilities
        all_targets.extend(target.detach().cpu().numpy())
        probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_outputs.extend(np.argmax(probs, axis=1))

        # Update meters
        batch_size = inp.shape[0]
        loss_meter.update(loss.item())
        acc1_meter.update(acc1.item(), n=batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'acc1': f"{acc1_meter.avg:.2f}"
        })

    # Convert to numpy arrays for metric calculations
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    # F1 Score
    metrics['f1'] = f1_score(all_targets, all_outputs, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_outputs)
    metrics['confusion_matrix'] = cm

    # For binary classification, calculate specificity
    if linear_classifier.module.num_labels == 2:
        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        # For multiclass, calculate specificity for each class using one-vs-rest
        specificities = []
        sensitivities = []
        for i in range(linear_classifier.module.num_labels):
            # True negatives: all samples that are not predicted as class i and are not class i
            tn = np.sum(np.logical_and(all_outputs != i, all_targets != i))
            # False positives: samples predicted as class i but are not class i
            fp = np.sum(np.logical_and(all_outputs == i, all_targets != i))
            # False negatives: samples not predicted as class i but are class i
            fn = np.sum(np.logical_and(all_outputs != i, all_targets == i))
            # True positives: samples predicted as class i and are class i
            tp = np.sum(np.logical_and(all_outputs == i, all_targets == i))
            
            # Specificity for current class
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
            
            # Sensitivity for current class
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sens)
        
        metrics['specificity'] = np.mean(specificities)
        metrics['sensitivity'] = np.mean(sensitivities)
    
    # For binary classification
    if linear_classifier.module.num_labels == 2:
        # AUROC - Area under ROC curve
        metrics['auroc'] = roc_auc_score(all_targets, all_probs[:, 1])
        
        # AUPRC - Area under Precision-Recall curve
        metrics['auprc'] = average_precision_score(all_targets, all_probs[:, 1])
    else:
        # For multiclass, use one-vs-rest approach
        auroc_list = []
        auprc_list = []
        
        # One-vs-rest approach for each class
        for i in range(linear_classifier.module.num_labels):
            # Create binary labels for current class (1 for current class, 0 for others)
            binary_targets = (all_targets == i).astype(int)
            class_probs = all_probs[:, i]
            
            # Calculate AUROC for current class
            try:
                auroc = roc_auc_score(binary_targets, class_probs)
                auroc_list.append(auroc)
            except ValueError:
                # This can happen if only one class present in y_true
                pass
                
            # Calculate AUPRC for current class  
            precision, recall, _ = precision_recall_curve(binary_targets, class_probs)
            auprc = auc(recall, precision)
            auprc_list.append(auprc)
            
        # Average metrics across all classes
        metrics['auroc'] = np.mean(auroc_list) if auroc_list else 0
        metrics['auprc'] = np.mean(auprc_list) if auprc_list else 0

    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg}, metrics


@torch.no_grad()
def validate_network_multi_view(val_loader, model, linear_classifier, n, avgpool, cfg):
    linear_classifier.eval()
    test_meter = TestMeter(
        len(val_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        args.num_labels,
        len(val_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
        )
    test_meter.iter_tic()
    all_targets = []
    all_outputs = []
    all_probs = []
    
    for cur_iter, (inp, target, sample_idx, meta) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Multi-view Eval"):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        test_meter.data_toc()

        # forward
        output = model(inp)
        output = linear_classifier(output)

        # Store results for metrics calculation
        all_targets.extend(target.detach().cpu().numpy())
        probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_outputs.extend(np.argmax(probs, axis=1))

        output = output.cpu()
        target = target.cpu()
        sample_idx = sample_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            output.detach(), target.detach(), sample_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    # F1 Score
    metrics['f1'] = f1_score(all_targets, all_outputs, average='weighted')
    
    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_outputs)
    
    # For binary classification
    if args.num_labels == 2:
        # AUROC - Area under ROC curve
        metrics['auroc'] = roc_auc_score(all_targets, all_probs[:, 1])
        
        # AUPRC - Area under Precision-Recall curve
        metrics['auprc'] = average_precision_score(all_targets, all_probs[:, 1])
    else:
        # For multiclass, use one-vs-rest approach
        auroc_list = []
        auprc_list = []
        
        # Calculate metrics for each class
        for i in range(args.num_labels):
            binary_targets = (all_targets == i).astype(int)
            class_probs = all_probs[:, i]
            
            try:
                auroc = roc_auc_score(binary_targets, class_probs)
                auroc_list.append(auroc)
            except ValueError:
                pass
                
            precision, recall, _ = precision_recall_curve(binary_targets, class_probs)
            auprc = auc(recall, precision)
            auprc_list.append(auprc)
            
        metrics['auroc'] = np.mean(auroc_list) if auroc_list else 0
        metrics['auprc'] = np.mean(auprc_list) if auprc_list else 0

    test_meter.finalize_metrics(ks=(1, ))
    return test_meter.stats, metrics


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'swin'],
                        help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="ucf101", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained_model_weights', default='polypdiag.pth', type=str, help='pre-trained weights')

    # Add sampling method arguments
    parser.add_argument('--train_sampling', default='random', type=str, 
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for training')
    parser.add_argument('--val_sampling', default='uniform', type=str, 
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for validation')
    parser.add_argument('--test_sampling', default='uniform', type=str, 
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for testing')
    parser.add_argument('--num_frames', default=32, type=int,
                        help='Number of frames to sample from each video')
                        
    # Add focal loss arguments
    parser.add_argument('--loss_function', default='cross_entropy', type=str,
                        choices=['cross_entropy', 'focal_loss'],
                        help='Loss function to use for training')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Gamma parameter for focal loss (focuses on hard examples)')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='Alpha parameter for focal loss (addresses class imbalance)')
    
    # Add seed argument - default to 42 for reproducibility
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    eval_finetune(args)