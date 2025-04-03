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

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, average_precision_score

from datasets import UCF101, HMDB51, Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config


def eval_finetune(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # ============ preparing data ... ============
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    if args.dataset == "ucf101":
        dataset_train = UCF101(cfg=config, mode="train", num_retries=10)
        dataset_val = UCF101(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
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
        ckpt = torch.load('kinetics400_vitb_ssl.pth', map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
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
        print(f"Metrics of the network on the {len(dataset_val)} test images:")
        print(f"F1 score: {metrics['f1'] * 100:.1f}%")
        print(f"AUROC: {metrics['auroc'] * 100:.1f}%")
        print(f"AUPRC: {metrics['auprc'] * 100:.1f}%")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
        
        exit(0)

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'lr': scaled_lr},
         {'params': linear_classifier.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0., "best_f1": 0., "best_auroc": 0., "best_auprc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]
    best_auroc = to_restore["best_auroc"]
    best_auprc = to_restore["best_auprc"]
    best_epoch = start_epoch - 1  # Track the epoch with the best performance

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(args, model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
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

            if metrics['f1'] > best_f1 and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "backbone_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": metrics['f1'],
                    "best_auroc": max(best_auroc, metrics['auroc']),
                    "best_auprc": max(best_auprc, metrics['auprc']),
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
                
                # Plot and save confusion matrix for best model
                plt.figure(figsize=(10, 8))
                sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix (Epoch {epoch})')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_epoch_{epoch}.png'))
                
                # Update best epoch tracking
                best_epoch = epoch

            # Update best metrics
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
            if metrics['auprc'] > best_auprc:
                best_auprc = metrics['auprc']
                
            print(f'Best metrics so far:')
            print(f'F1: {best_f1 * 100:.1f}%')
            print(f'AUROC: {best_auroc * 100:.1f}%')
            print(f'AUPRC: {best_auprc * 100:.1f}%')

    # After all epochs, load the best model and generate final metrics and confusion matrix
    if not args.test and utils.is_main_process():
        print("\nTraining complete. Loading best model for final evaluation...")
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth.tar"),
            backbone_state_dict=model,
            state_dict=linear_classifier,
        )
        
        final_stats, final_metrics = validate_network(val_loader, model, linear_classifier, 
                                                   args.n_last_blocks, args.avgpool_patchtokens)
        
        print(f"Final metrics with best model (from epoch {best_epoch}):")
        print(f"F1 score: {final_metrics['f1'] * 100:.1f}%")
        print(f"AUROC: {final_metrics['auroc'] * 100:.1f}%")
        print(f"AUPRC: {final_metrics['auprc'] * 100:.1f}%")
        
        # Plot and save confusion matrix for best overall model
        plt.figure(figsize=(10, 8))
        sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Confusion Matrix (Best Model from Epoch {best_epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(args.output_dir, 'final_confusion_matrix.png'))
        print(f"Final confusion matrix saved to {os.path.join(args.output_dir, 'final_confusion_matrix.png')}")


def train(args, model, linear_classifier, optimizer, loader, epoch, n, avgpool):
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

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

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
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_outputs)
    
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

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    eval_finetune(args)