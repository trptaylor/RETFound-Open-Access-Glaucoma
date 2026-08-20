import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):

        samples = batch[0]
        targets = batch[-1]

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        loss /= accum_iter

        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()

    true_onehot, pred_onehot = [], []
    true_labels, pred_labels = [], []
    pred_softmax = []
    image_names = []
    predict_results = []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):

        if args.predict:
            images = batch[0].to(device, non_blocking=True)
            names = batch[1]
        else:
            images = batch[0].to(device, non_blocking=True)
            names = batch[1]
            target = batch[2].to(device, non_blocking=True)

            image_names.extend(names)
            target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.cuda.amp.autocast():
            output = model(images)
            if not args.predict:
                loss = criterion(output, target)

        output_ = torch.softmax(output, dim=1)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label, num_classes=num_class)

        if not args.predict:
            metric_logger.update(loss=loss.item())

            true_onehot.extend(target_onehot.cpu().numpy())
            pred_onehot.extend(output_onehot.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

        pred_labels.extend(output_label.cpu().numpy())
        pred_softmax.extend(output_.cpu().numpy())

        if args.predict:
            for n, p, s in zip(names,
                               output_label.cpu().numpy(),
                               output_.cpu().numpy()):
                predict_results.append((os.path.basename(n), p, s))

    # ========================
    # Predict mode exit early
    # ========================
    if args.predict and misc.is_main_process():
        pred_path = os.path.join(args.output_dir, args.task, "predictions.csv")

        with open(pred_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_name', 'predicted_label', 'softmax'])
            writer.writerows(predict_results)

        print(f"✅ Predictions saved to {pred_path}")
        return {}, 0

    # ========================
    # Normal metric computation
    # ========================
    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')

    score = (f1 + roc_auc + kappa) / 3

    if "loss" in metric_logger.meters:
        print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}')

    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score
