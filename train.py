from __future__ import absolute_import

# system lib
import os
import time
import sys
import argparse
# numerical libs
import random

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchinfo import summary
# model_ckpts
# from thop import profile
from utils.util import WeightedAverageMeter, AverageMeter, ProgressMeter, accuracy, parse_gpus, meanSquaredError, seed_everything
from utils.checkpoint import save_checkpoint, load_checkpoint
from models.multimodal import create_net, create_multimodal_net
from models.CopulaLoss import CopulaLoss
from models.CopulaLoss import copula_loss_instantiation
from Data import patient_train_val_split, process_image_cov_resp_report, train_val_dataloader
import matplotlib.pyplot as plt
from models.WeightedMSE import WeightedMSE
from models.WeightedBCE import WeightedBCE, FocalBCE
import subprocess
import math


def adjust_learning_rate(optimizer, epoch, base_lr, warmup=False):
    """Adjust the learning rate"""
    if epoch <= 20:
        # lr = 0.00001 if warmup and epoch == 0 else args.base_lr
        lr = 0.00001 if warmup else base_lr
    elif epoch <= 60:
        lr = base_lr * 0.1
    elif epoch <= 80:
        lr = base_lr * 0.01
    elif epoch <= 100:
        lr = base_lr * 0.001
    else:
        lr = base_lr * 0.0001

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_learning_rate_mm(optimizer, epoch, base_lr, warmup=False):
    """Adjust the learning rate"""
    if epoch <= 5:
        # lr = 0.00001 if warmup and epoch == 0 else args.base_lr
        lr = 0.00001 if warmup else base_lr
    elif epoch <= 10:
        lr = base_lr
    elif epoch <= 60:
        lr = base_lr * 0.1
    elif epoch <= 100:
        lr = base_lr * 0.01
    else:
        lr = base_lr * 0.001

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_learning_rate_mm_3stage(optimizer, epoch, base_lr, warmup=False, stage=1):
    """Adjust the learning rate"""
    # 20 epoch, origin: 40 epoch
    if stage == 1:
        if epoch <= 5:
            lr = 0.00001 if warmup else base_lr
        elif epoch <= 10:
            lr = base_lr
        else:
            lr = 0.1 * base_lr
    # 10 epoch, origin: 40 epoch and lr = 0.1 * base_lr
    elif stage == 2:
        if epoch <= 25:
            lr = 0.01 * base_lr  # origin 0.1 * base_lr
        else:
            lr = 0.001 * base_lr  # origin 0.1 * base_lr
    # 10 epoch, origin: 40 epoch and lr = 0.1 * base_lr
    else:
        '''
        if epoch <= 35:
            lr = 0.01 * base_lr  # origin 0.1 * base_lr
        else:
            lr = 0.001 * base_lr  # origin 0.01 * base_lr
        '''
        if epoch <= 32:
            lr = 0.001 * base_lr  # origin 0.1 * base_lr
        elif epoch <= 35:
            lr = 0.01 * base_lr  # origin 0.01 * base_lr
        else:
            lr = 0.001 * base_lr  # origin 0.01 * base_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_learning_rate_mm_3stage_cos(optimizer, epoch, base_lr, stage_epoch=[20, 20, 20], warmup=False):
    stage_boundaries = [stage_epoch[0],
                        stage_epoch[0] + stage_epoch[1],
                        sum(stage_epoch)]
    if epoch <= stage_boundaries[0]:
        stage = 1
        local_epoch = epoch - 1
        total_stage_epochs = stage_epoch[0]
    elif epoch <= stage_boundaries[1]:
        stage = 2
        local_epoch = epoch - stage_boundaries[0] - 1
        total_stage_epochs = stage_epoch[1]
    else:
        stage = 3
        local_epoch = epoch - stage_boundaries[1] - 1
        total_stage_epochs = stage_epoch[2]

    if stage == 1:
        lr_max = base_lr
        lr_min = 0.1 * base_lr
    elif stage == 2:
        lr_max = 0.01 * base_lr
        lr_min = 0.001 * base_lr
    else:
        lr_max = 0.0005 * base_lr
        lr_min = 0.0001 * base_lr
    # stage 1: warmup + cosA
    # stage 2: cosA
    # stage 3: cosA
    '''
    if warmup and local_epoch < 5 and stage == 1:
        lr = lr_max * (local_epoch + 1) / 5.0

    else:
        if stage == 1:
            progress = local_epoch / max(1, total_stage_epochs - 5)
        else:
            progress = local_epoch / max(1, total_stage_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    '''
    progress = local_epoch / max(1, total_stage_epochs)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_classification_metrics(y_pred, y_true, threshold=0.5):
    y_pred_label = (y_pred >= threshold).float()

    correct = (y_pred_label == y_true).float().sum()
    accuracy = correct / y_true.numel()

    TP = ((y_pred_label == 1) & (y_true == 1)).sum().float()
    FN = ((y_pred_label == 0) & (y_true == 1)).sum().float()
    sensitivity = TP / (TP + FN + 1e-8)

    return accuracy, sensitivity, TP + FN + 1e-8


def train(net, optimizer, epoch, data_loader, args, copulaLoss=None):
    learning_rate = optimizer.param_groups[0]["lr"]

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    if args.classification:
        losses = WeightedAverageMeter('Loss', args.res_dim, device=args.device, fmt=':.4f')
    else:
        losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        acc1_meter = AverageMeter('Acc1', ':4.6f')
        sens1_meter = AverageMeter('Sens1', ':4.6f')
        acc2_meter = AverageMeter('Acc2', ':4.6f')
        sens2_meter = AverageMeter('Sens2', ':4.6f')
        acc3_meter = AverageMeter('Acc3', ':4.6f')
        sens3_meter = AverageMeter('Sens3', ':4.6f')
        progress_list = [batch_time, data_time, losses, acc1_meter, sens1_meter, acc2_meter, sens2_meter, acc3_meter, sens3_meter]
        if args.DLCO_flag:
            acc4_meter = AverageMeter('Acc4', ':4.6f')
            sens4_meter = AverageMeter('Sens4', ':4.6f')
            progress_list.extend([acc4_meter, sens4_meter])
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter
    else:
        mse_meter = AverageMeter('MSE', ':4.6f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, mse_meter],
            prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter

    net.train()  # Set the model to training mode

    tic = time.time()
    for batch_idx, (image, x, report, reasoning, reasoning_mask, target, diag) in enumerate(data_loader):

        image, x, target, diag = image.to(args.device, non_blocking=True), \
                           x.to(args.device, non_blocking=True), \
                           target.to(args.device, non_blocking=True), \
                                 diag.to(args.device, non_blocking=True)
        if args.CT_report_flag:
            report = report.to(args.device, non_blocking=True)
        if args.reasoning_flag:
            reasoning = reasoning.to(args.device, non_blocking=True)
            reasoning_mask = reasoning_mask.to(args.device, non_blocking=True)

        data_time.update(time.time() - tic)

        optimizer.zero_grad()  # Clear gradients for next training step

        output = net(image, x, report=report, reasoning=reasoning, reasoning_mask=reasoning_mask)  # Forward pass
        if copulaLoss:  # The classification version need to be coded.
            loss = copulaLoss(target, output)
            total_weight = image.size(0)
        else:
            if args.LLM_predict_flag:
                if args.classification:
                    WBCE = WeightedBCE(x[:, 4:], diag, factor=args.LLM_factor, DLCO_flag=args.DLCO_flag,
                                       device=args.device, alpha=args.focal_alpha, gamma=args.focal_gamma)
                    losses_split = WBCE.weighted_loss
                    total_weight = WBCE.total_weight
                    loss = WBCE(output, diag)
                else:
                    WMSE = WeightedMSE(x[:, 4:], factor=args.LLM_factor, DLCO_flag=args.DLCO_flag,
                                       device=args.device)
                    total_weight = WMSE.total_weight.item()
                    loss = WMSE(output, target)
            else:
                if args.classification:
                    FBCE = FocalBCE(diag, alpha=args.focal_alpha, DLCO_flag=args.DLCO_flag, device=args.device, gamma=args.focal_gamma)
                    losses_split = FBCE.weighted_loss
                    total_weight = FBCE.total_weight
                    loss = FBCE(output, diag)
                    # total_weight = args.res_dim * image.size(0)
                else:
                    loss = F.mse_loss(output, target, reduction='mean')  # Compute loss
                    total_weight = args.res_dim * image.size(0)
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters
        if args.classification:
            losses.update(losses_split, total_weight)
        else:
            losses.update(loss.item(), total_weight)
        # Calculate MSE between predicted values and target values
        if args.classification:
            acc1, sens1, pos_n1 = compute_classification_metrics(output[:, 0], diag[:, 0], threshold=0.5)
            acc1_meter.update(acc1.item(), image.size(0))
            sens1_meter.update(sens1.item(), pos_n1.item())

            acc2, sens2, pos_n2 = compute_classification_metrics(output[:, 1], diag[:, 1], threshold=0.5)
            acc2_meter.update(acc2.item(), image.size(0))
            sens2_meter.update(sens2.item(), pos_n2.item())

            acc3, sens3, pos_n3 = compute_classification_metrics(output[:, 2], diag[:, 2], threshold=0.5)
            acc3_meter.update(acc3.item(), image.size(0))
            sens3_meter.update(sens3.item(), pos_n3.item())
            if args.DLCO_flag:
                acc4, sens4, pos_n4 = compute_classification_metrics(output[:, 3], diag[:, 3], threshold=0.5)
                acc4_meter.update(acc4.item(), image.size(0))
                sens4_meter.update(sens4.item(), pos_n4.item())
        else:
            mse_value = F.mse_loss(output, target, reduction='mean')  # Compute Mean Squared Error
            mse_meter.update(mse_value.item(), args.res_dim * image.size(0))  # Update the MSE metric

        batch_time.update(time.time() - tic)
        tic = time.time()
        if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
            epoch_msg = progress.get_message(batch_idx + 1)
            print(epoch_msg)

            args.log_file.write(epoch_msg + "\n")

    if args.classification:
        loss_avg = losses.avg.cpu().item()

        if args.DLCO_flag:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, acc4_meter.avg, sens4_meter.avg, loss_avg
        else:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, 0, 0, loss_avg
    else:
        return mse_meter.avg


def train_3stage(net, optimizer, epoch, data_loader, args, stage=1):
    learning_rate = optimizer.param_groups[0]["lr"]

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    if args.classification:
        losses = WeightedAverageMeter('Loss', args.res_dim, device=args.device, fmt=':.4f')
    else:
        losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        acc1_meter = AverageMeter('Acc1', ':4.6f')
        sens1_meter = AverageMeter('Sens1', ':4.6f')
        acc2_meter = AverageMeter('Acc2', ':4.6f')
        sens2_meter = AverageMeter('Sens2', ':4.6f')
        acc3_meter = AverageMeter('Acc3', ':4.6f')
        sens3_meter = AverageMeter('Sens3', ':4.6f')
        progress_list = [batch_time, data_time, losses, acc1_meter, sens1_meter, acc2_meter, sens2_meter, acc3_meter, sens3_meter]
        if args.DLCO_flag:
            acc4_meter = AverageMeter('Acc4', ':4.6f')
            sens4_meter = AverageMeter('Sens4', ':4.6f')
            progress_list.extend([acc4_meter, sens4_meter])
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter
    else:
        mse_meter = AverageMeter('MSE', ':4.6f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, mse_meter],
            prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter

    net.train()  # Set the model to training mode

    tic = time.time()
    for batch_idx, (image, x, report, reasoning, reasoning_mask, target, diag) in enumerate(data_loader):

        image, x, target, diag = image.to(args.device, non_blocking=True), \
                           x.to(args.device, non_blocking=True), \
                           target.to(args.device, non_blocking=True), \
                                 diag.to(args.device, non_blocking=True)

        if args.CT_report_flag:
            report = report.to(args.device, non_blocking=True)
        if args.reasoning_flag:
            reasoning = reasoning.to(args.device, non_blocking=True)
            reasoning_mask = reasoning_mask.to(args.device, non_blocking=True)

        data_time.update(time.time() - tic)

        optimizer.zero_grad()  # Clear gradients for next training step

        output = net(image, x, report=report, reasoning=reasoning, reasoning_mask=reasoning_mask)  # Forward pass

        if args.LLM_predict_flag:
            if args.classification:
                if args.stage3_train:
                    if stage == 1:  # Text Stage
                        factor = args.LLM_factor
                        fac_low_cc = False
                    elif stage == 2:  # Image Stage
                        factor = args.LLM_factor
                        fac_low_cc = True
                    else:  # Joint Stage
                        if args.DLCO_flag:
                            factor = [1.0, 1.0, 1.0, 1.0]
                        else:
                            factor = [1.0, 1.0, 1.0]
                        fac_low_cc = False
                else:
                    factor = args.LLM_factor
                    fac_low_cc = True
                WBCE = WeightedBCE(x[:, 4:], diag, factor=factor, DLCO_flag=args.DLCO_flag,
                                   device=args.device, alpha=args.focal_alpha, fac_low_cc=fac_low_cc, gamma=args.focal_gamma)
                losses_split = WBCE.weighted_loss
                total_weight = WBCE.total_weight
                loss = WBCE(output, diag)
            else:
                WMSE = WeightedMSE(x[:, 4:], factor=args.LLM_factor, DLCO_flag=args.DLCO_flag,
                                   device=args.device)
                total_weight = WMSE.total_weight.item()
                loss = WMSE(output, target)
        else:
            if args.classification:
                FBCE = FocalBCE(diag, alpha=args.focal_alpha, DLCO_flag=args.DLCO_flag, device=args.device, gamma=args.focal_gamma)
                losses_split = FBCE.weighted_loss
                total_weight = FBCE.total_weight
                loss = FBCE(output, diag)
                # loss = F.binary_cross_entropy(output, diag, reduction='mean')
                # total_weight = args.res_dim * image.size(0)
            else:
                loss = F.mse_loss(output, target, reduction='mean')  # Compute loss
                total_weight = args.res_dim * image.size(0)

        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters
        if args.classification:
            losses.update(losses_split, total_weight)
        else:
            losses.update(loss.item(), total_weight)
        # Calculate MSE between predicted values and target values
        if args.classification:
            acc1, sens1, pos_n1 = compute_classification_metrics(output[:, 0], diag[:, 0], threshold=0.5)
            acc1_meter.update(acc1.item(), image.size(0))
            sens1_meter.update(sens1.item(), pos_n1.item())

            acc2, sens2, pos_n2 = compute_classification_metrics(output[:, 1], diag[:, 1], threshold=0.5)
            acc2_meter.update(acc2.item(), image.size(0))
            sens2_meter.update(sens2.item(), pos_n2.item())

            acc3, sens3, pos_n3 = compute_classification_metrics(output[:, 2], diag[:, 2], threshold=0.5)
            acc3_meter.update(acc3.item(), image.size(0))
            sens3_meter.update(sens3.item(), pos_n3.item())
            if args.DLCO_flag:
                acc4, sens4, pos_n4 = compute_classification_metrics(output[:, 3], diag[:, 3], threshold=0.5)
                acc4_meter.update(acc4.item(), image.size(0))
                sens4_meter.update(sens4.item(), pos_n4.item())
        else:
            mse_value = F.mse_loss(output, target, reduction='mean')  # Compute Mean Squared Error
            mse_meter.update(mse_value.item(), args.res_dim * image.size(0))  # Update the MSE metric

        batch_time.update(time.time() - tic)
        tic = time.time()
        if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
            epoch_msg = progress.get_message(batch_idx + 1)
            print(epoch_msg)

            args.log_file.write(epoch_msg + "\n")

    if args.classification:
        loss_avg = losses.avg.cpu().item()
        if args.DLCO_flag:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, acc4_meter.avg, sens4_meter.avg, loss_avg
        else:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, 0, 0, loss_avg
    else:
        return mse_meter.avg


def validate(net, epoch, data_loader, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        losses = WeightedAverageMeter('Loss', args.res_dim, device=args.device, fmt=':.4f')
    else:
        losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        acc1_meter = AverageMeter('Acc1', ':4.6f')
        sens1_meter = AverageMeter('Sens1', ':4.6f')
        acc2_meter = AverageMeter('Acc2', ':4.6f')
        sens2_meter = AverageMeter('Sens2', ':4.6f')
        acc3_meter = AverageMeter('Acc3', ':4.6f')
        sens3_meter = AverageMeter('Sens3', ':4.6f')
        progress_list = [batch_time, data_time, losses, acc1_meter, sens1_meter, acc2_meter, sens2_meter, acc3_meter, sens3_meter]
        if args.DLCO_flag:
            acc4_meter = AverageMeter('Acc4', ':4.6f')
            sens4_meter = AverageMeter('Sens4', ':4.6f')
            progress_list.extend([acc4_meter, sens4_meter])
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Epoch (Valid LR {:6.6f}): [{}] ".format(0, epoch))  # Initialize ProgressMeter
    else:
        mse_meter = AverageMeter('MSE', ':4.6f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, mse_meter],
            prefix="Epoch (Valid LR {:6.6f}): [{}] ".format(0, epoch))  # Initialize ProgressMeter

    net.eval()

    with torch.no_grad():
        tic = time.time()
        for batch_idx, (image, x, report, reasoning, reasoning_mask, target, diag) in enumerate(data_loader):

            image, x, target, diag = image.to(args.device, non_blocking=True), \
                               x.to(args.device, non_blocking=True), \
                               target.to(args.device, non_blocking=True), \
                                     diag.to(args.device, non_blocking=True)
            if args.CT_report_flag:
                report = report.to(args.device, non_blocking=True)
            if args.reasoning_flag:
                reasoning = reasoning.to(args.device, non_blocking=True)
                reasoning_mask = reasoning_mask.to(args.device, non_blocking=True)
            data_time.update(time.time() - tic)
            output = net(image, x, report=report, reasoning=reasoning, reasoning_mask=reasoning_mask)  # Forward pass

            if args.classification:
                FBCE = FocalBCE(diag, alpha=args.val_focal_alpha, DLCO_flag=args.DLCO_flag, device=args.device, gamma=args.focal_gamma)
                losses_split = FBCE.weighted_loss
                total_weight = FBCE.total_weight
                loss = FBCE(output, diag)
                # loss = F.binary_cross_entropy(output, diag, reduction='mean')
                # total_weight = args.res_dim * image.size(0)
            else:
                loss = F.mse_loss(output, target, reduction='mean')  # Compute loss
                total_weight = image.size(0) * args.res_dim

            if args.classification:
                losses.update(losses_split, total_weight)
            else:
                losses.update(loss.item(), total_weight)
            # Calculate MSE between predicted values and target values
            if args.classification:
                acc1, sens1, pos_n1 = compute_classification_metrics(output[:, 0], diag[:, 0], threshold=0.5)
                acc1_meter.update(acc1.item(), image.size(0))
                sens1_meter.update(sens1.item(), pos_n1.item())

                acc2, sens2, pos_n2 = compute_classification_metrics(output[:, 1], diag[:, 1], threshold=0.5)
                acc2_meter.update(acc2.item(), image.size(0))
                sens2_meter.update(sens2.item(), pos_n2.item())

                acc3, sens3, pos_n3 = compute_classification_metrics(output[:, 2], diag[:, 2], threshold=0.5)
                acc3_meter.update(acc3.item(), image.size(0))
                sens3_meter.update(sens3.item(), pos_n3.item())

                if args.DLCO_flag:
                    acc4, sens4, pos_n4 = compute_classification_metrics(output[:, 3], diag[:, 3], threshold=0.5)
                    acc4_meter.update(acc4.item(), image.size(0))
                    sens4_meter.update(sens4.item(), pos_n4.item())
            else:
                mse_value = F.mse_loss(output, target, reduction='mean')  # Compute Mean Squared Error
                mse_meter.update(mse_value.item(), image.size(0))  # Update the MSE metric

            batch_time.update(time.time() - tic)
            tic = time.time()
            if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
                epoch_msg = progress.get_message(batch_idx + 1)
                print(epoch_msg)

                args.log_file.write(epoch_msg + "\n")
    if args.classification:
        loss_avg = losses.avg.cpu().item()
        if args.DLCO_flag:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, acc4_meter.avg, sens4_meter.avg, loss_avg
        else:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, 0, 0, loss_avg
    else:
        return mse_meter.avg


def test(net, epoch, data_loader, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        losses = WeightedAverageMeter('Loss', args.res_dim, device=args.device, fmt=':.4f')
    else:
        losses = AverageMeter('Loss', ':.4f')
    if args.classification:
        acc1_meter = AverageMeter('Acc1', ':4.6f')
        sens1_meter = AverageMeter('Sens1', ':4.6f')
        acc2_meter = AverageMeter('Acc2', ':4.6f')
        sens2_meter = AverageMeter('Sens2', ':4.6f')
        acc3_meter = AverageMeter('Acc3', ':4.6f')
        sens3_meter = AverageMeter('Sens3', ':4.6f')
        progress_list = [batch_time, data_time, losses, acc1_meter, sens1_meter, acc2_meter, sens2_meter, acc3_meter, sens3_meter]
        if args.DLCO_flag:
            acc4_meter = AverageMeter('Acc4', ':4.6f')
            sens4_meter = AverageMeter('Sens4', ':4.6f')
            progress_list.extend([acc4_meter, sens4_meter])
        progress = ProgressMeter(
            len(data_loader),
            progress_list,
            prefix="Epoch (Test LR {:6.6f}): [{}] ".format(0, epoch))  # Initialize ProgressMeter
    else:
        mse_meter = AverageMeter('MSE', ':4.6f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, mse_meter],
            prefix="Epoch (Test LR {:6.6f}): [{}] ".format(0, epoch))  # Initialize ProgressMeter

    net.eval()

    with torch.no_grad():
        tic = time.time()
        for batch_idx, (image, x, report, reasoning, reasoning_mask, target, diag) in enumerate(data_loader):

            image, x, target, diag = image.to(args.device, non_blocking=True), \
                               x.to(args.device, non_blocking=True), \
                               target.to(args.device, non_blocking=True), \
                                     diag.to(args.device, non_blocking=True)
            if args.CT_report_flag:
                report = report.to(args.device, non_blocking=True)
            if args.reasoning_flag:
                reasoning = reasoning.to(args.device, non_blocking=True)
                reasoning_mask = reasoning_mask.to(args.device, non_blocking=True)
            data_time.update(time.time() - tic)
            output = net(image, x, report=report, reasoning=reasoning, reasoning_mask=reasoning_mask)  # Forward pass

            if args.classification:
                FBCE = FocalBCE(diag, alpha=args.val_focal_alpha, DLCO_flag=args.DLCO_flag, device=args.device, gamma=args.focal_gamma)
                losses_split = FBCE.weighted_loss
                total_weight = FBCE.total_weight
                loss = FBCE(output, diag)
                # loss = F.binary_cross_entropy(output, diag, reduction='mean')
                # total_weight = args.res_dim * image.size(0)
            else:
                loss = F.mse_loss(output, target, reduction='mean')  # Compute loss
                total_weight = image.size(0) * args.res_dim

            if args.classification:
                losses.update(losses_split, total_weight)
            else:
                losses.update(loss.item(), total_weight)
            # Calculate MSE between predicted values and target values
            if args.classification:
                acc1, sens1, pos_n1 = compute_classification_metrics(output[:, 0], diag[:, 0], threshold=0.5)
                acc1_meter.update(acc1.item(), image.size(0))
                sens1_meter.update(sens1.item(), pos_n1.item())

                acc2, sens2, pos_n2 = compute_classification_metrics(output[:, 1], diag[:, 1], threshold=0.5)
                acc2_meter.update(acc2.item(), image.size(0))
                sens2_meter.update(sens2.item(), pos_n2.item())

                acc3, sens3, pos_n3 = compute_classification_metrics(output[:, 2], diag[:, 2], threshold=0.5)
                acc3_meter.update(acc3.item(), image.size(0))
                sens3_meter.update(sens3.item(), pos_n3.item())

                if args.DLCO_flag:
                    acc4, sens4, pos_n4 = compute_classification_metrics(output[:, 3], diag[:, 3], threshold=0.5)
                    acc4_meter.update(acc4.item(), image.size(0))
                    sens4_meter.update(sens4.item(), pos_n4.item())
            else:
                mse_value = F.mse_loss(output, target, reduction='mean')  # Compute Mean Squared Error
                mse_meter.update(mse_value.item(), image.size(0))  # Update the MSE metric

            batch_time.update(time.time() - tic)
            tic = time.time()
            if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
                epoch_msg = progress.get_message(batch_idx + 1)
                print(epoch_msg)

                args.log_file.write(epoch_msg + "\n")
    if args.classification:
        loss_avg = losses.avg.cpu().item()
        if args.DLCO_flag:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, acc4_meter.avg, sens4_meter.avg, loss_avg
        else:
            return acc1_meter.avg, sens1_meter.avg, acc2_meter.avg, sens2_meter.avg, acc3_meter.avg, sens3_meter.avg, 0, 0, loss_avg
    else:
        return mse_meter.avg


def main(args, train_loader, val_loader, test_loader_CT, test_loader_MM):
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        cudnn.benchmark = True
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    net = create_multimodal_net(args)

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.base_lr,
                              momentum=args.beta1, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.base_lr,
                              betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    if args.resume:
        net, optimizer, best_mse, start_epoch = load_checkpoint(args, net, optimizer)
    else:
        start_epoch = 0
        best_mse = 1e8
        best_bce = 1e8
        all_best_bce = 1e8
        best_acc1 = 0
        best_acc2 = 0
        best_acc3 = 0
        best_acc4 = 0
        best_sens1 = 0
        best_sens2 = 0
        best_sens3 = 0
        best_sens4 = 0

    args.log_file.write("Network - " + args.arch + "\n")
    for key, val in vars(args).items():
        args.log_file.write("{:16} {}".format(key, val) + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    # multi-GPUs
    if len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, args.gpu_ids)

    if args.CT_report_flag or args.reasoning_flag:
        image_model_state_dict = torch.load(args.image_feature_net_para_path)['state_dict']
        feature_extractor_state_dict = {k.replace("feature_extractor.", ""): v
                                        for k, v in image_model_state_dict.items() if
                                        k.startswith("feature_extractor.")}
        net.feature_extractor.load_state_dict(feature_extractor_state_dict)
        if args.CT_report_flag and not args.reasoning_flag:
            for param in net.feature_extractor.parameters():
                param.requires_grad = True
        else:
            for param in net.feature_extractor.parameters():
                param.requires_grad = False
    else:
        for param in net.feature_extractor.parameters():
            param.requires_grad = True

    net.to(args.device)

    summary(net)
    if args.classification:
        train_bces = []
        val_bces = []
        test_bces_CT = []
        test_bces_MM = []

    stage = 1 if args.stage3_train else 0
    if args.stage3_train:
        print(f'Training stage{stage} start!')
        stage_epoch = [10, 10, 10]  # origin: [40, 40, 1000]
        stage_shift = False
    for epoch in range(start_epoch, args.num_epoch):
        if args.reasoning_flag:
            if args.stage3_train:
                if epoch >= sum(stage_epoch[:stage]):
                    stage += 1
                    stage_shift = True
                    best_bce = 1e8
                # adjust_learning_rate_mm_3stage(optimizer, epoch, args.base_lr, args.warmup, stage=stage)
                adjust_learning_rate_mm_3stage_cos(optimizer, epoch + 1, args.base_lr, stage_epoch=stage_epoch, warmup=args.warmup)
            else:
                adjust_learning_rate_mm(optimizer, epoch, args.base_lr, args.warmup)
        else:
            if args.CT_report_flag:
                adjust_learning_rate_mm(optimizer, epoch, args.base_lr, args.warmup)
                if epoch > 10:
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = True
            else:
                adjust_learning_rate(optimizer, epoch, args.base_lr, args.warmup)

        if args.reasoning_flag:
            if args.stage3_train:
                if stage == 2 and stage_shift:
                    print(f'Training stage{stage} preapare...')
                    # print(f'Loading best model...')
                    print(f'Loading stage{stage - 1} best model...')
                    checkpoint = torch.load(args.ckpt + f'/model_best_checkpoint_stage{stage-1}.pth.tar')
                    # checkpoint = torch.load(args.ckpt + '/model_best_checkpoint.pth.tar')
                    net.load_state_dict(checkpoint['state_dict'], strict=True)
                    net.to(args.device)
                    print('Loading complete.')
                    for param in net.parameters():
                        param.requires_grad = False
                    # Unfreeze all layers about image
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = True

                    for param in net.ln_img.parameters():
                        param.requires_grad = True

                    for param in net.img_self_att.parameters():
                        param.requires_grad = True

                    for param in net.conv1d.parameters():
                        param.requires_grad = True

                    for param in net.mlp_layers[0].parameters():
                        param.requires_grad = True
                    for param in net.mlp_layers[1].parameters():
                        param.requires_grad = True
                    for param in net.mlp_layers[2].parameters():
                        param.requires_grad = True

                    stage_shift = False
                    summary(net)
                    print(f'Training stage{stage} start!')
                if stage == 3 and stage_shift:
                    print(f'Training stage{stage} preapare...')
                    print(f'Loading stage{stage-1} best model...')
                    checkpoint = torch.load(args.ckpt + f'/model_best_checkpoint_stage{stage - 1}.pth.tar')
                    # checkpoint = torch.load(args.ckpt + '/model_best_checkpoint.pth.tar')
                    net.load_state_dict(checkpoint['state_dict'], strict=True)
                    net.to(args.device)
                    print('Loading complete.')
                    for param in net.parameters():
                        param.requires_grad = True
                    stage_shift = False
                    summary(net)
                    print(f'Training stage{stage} start!')
            else:
                if epoch >= stage_epoch[0] and not stage_shift:
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = True
                    stage_shift = True

        print('lr now: %1.8f' % optimizer.param_groups[0]['lr'])
        args.log_file.write('lr now: %1.8f' % optimizer.param_groups[0]['lr'])
        if not os.path.exists(args.ckpt + '/MSE_log/'):
            os.makedirs(args.ckpt + '/MSE_log/')

        if args.reasoning_flag and args.stage3_train:
            train_epoch_acc1, train_epoch_sens1, train_epoch_acc2, train_epoch_sens2, train_epoch_acc3, train_epoch_sens3, train_epoch_acc4, train_epoch_sens4, train_epoch_bce = train_3stage(
                net, optimizer, epoch, train_loader, args, stage=stage)
        else:
            train_epoch_acc1, train_epoch_sens1, train_epoch_acc2, train_epoch_sens2, train_epoch_acc3, train_epoch_sens3, train_epoch_acc4, train_epoch_sens4, train_epoch_bce = train(net, optimizer, epoch, train_loader, args)
        val_epoch_acc1, val_epoch_sens1, val_epoch_acc2, val_epoch_sens2, val_epoch_acc3, val_epoch_sens3, val_epoch_acc4, val_epoch_sens4, val_epoch_bce = validate(net, epoch, val_loader, args)
        test_CT_epoch_acc1, test_CT_epoch_sens1, test_CT_epoch_acc2, test_CT_epoch_sens2, test_CT_epoch_acc3, test_CT_epoch_sens3, test_CT_epoch_acc4, test_CT_epoch_sens4, test_CT_epoch_bce = test(net, epoch, test_loader_CT, args)
        test_MM_epoch_acc1, test_MM_epoch_sens1, test_MM_epoch_acc2, test_MM_epoch_sens2, test_MM_epoch_acc3, test_MM_epoch_sens3, test_MM_epoch_acc4, test_MM_epoch_sens4, test_MM_epoch_bce = test(net, epoch, test_loader_MM, args)

        train_bces.append(train_epoch_bce)
        val_bces.append(val_epoch_bce)
        test_bces_CT.append(test_CT_epoch_bce)
        test_bces_MM.append(test_MM_epoch_bce)

        np.save(args.ckpt + '/MSE_log/train_bces.npy', np.array(train_bces))
        np.save(args.ckpt + '/MSE_log/val_bces.npy', np.array(val_bces))
        np.save(args.ckpt + '/MSE_log/test_bces_CT.npy', np.array(test_bces_CT))
        np.save(args.ckpt + '/MSE_log/test_bces_MM.npy', np.array(test_bces_MM))

        is_best = val_epoch_bce < best_bce

        all_is_best = val_epoch_bce < all_best_bce

        best_bce = min(val_epoch_bce, best_bce)
        all_best_bce = min(val_epoch_bce, all_best_bce)
        best_acc1 = max(val_epoch_acc1, best_acc1)
        best_acc2 = max(val_epoch_acc2, best_acc2)
        best_acc3 = max(val_epoch_acc3, best_acc3)
        if args.DLCO_flag:
            best_acc4 = max(val_epoch_acc4, best_acc4)
        best_sens1 = max(val_epoch_sens1, best_sens1)
        best_sens2 = max(val_epoch_sens2, best_sens2)
        best_sens3 = max(val_epoch_sens3, best_sens3)
        if args.DLCO_flag:
            best_sens4 = max(val_epoch_sens4, best_sens4)
        print("best bce: ", best_bce)
        print("best acc 1: ", best_acc1)
        print("best acc 2: ", best_acc2)
        print("best acc 3: ", best_acc3)
        if args.DLCO_flag:
            print("best acc 4: ", best_acc4)
        print("best sens 1: ", best_sens1)
        print("best sens 2: ", best_sens2)
        print("best sens 3: ", best_sens3)
        if args.DLCO_flag:
            print("best sens 4: ", best_sens4)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": net.cpu().state_dict(),  # net.module.cpu().state_dict() will error for no data parallel
            "best_mse": best_mse,
            "best_bce": best_bce,
            "best_acc1": best_acc1,
            "best_acc2": best_acc2,
            "best_acc3": best_acc3,
            "best_acc4": best_acc4,
            "best_sens1": best_sens1,
            "best_sens2": best_sens2,
            "best_sens3": best_sens3,
            "best_sens4": best_sens4,
            "optimizer": optimizer.state_dict(),
        }, is_best, epoch + 1, save_path=args.ckpt, stage=stage)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": net.cpu().state_dict(),  # net.module.cpu().state_dict() will error for no data parallel
            "best_mse": best_mse,
            "best_bce": best_bce,
            "best_acc1": best_acc1,
            "best_acc2": best_acc2,
            "best_acc3": best_acc3,
            "best_acc4": best_acc4,
            "best_sens1": best_sens1,
            "best_sens2": best_sens2,
            "best_sens3": best_sens3,
            "best_sens4": best_sens4,
            "optimizer": optimizer.state_dict(),
        }, all_is_best, epoch + 1, save_path=args.ckpt, stage=0)

        net.to(args.device)

        args.log_file.write("--------------------------------------------------" + "\n")

    if not os.path.exists(args.ckpt + '/MSE_log/'):
        os.makedirs(args.ckpt + '/MSE_log/')

    args.log_file.write("--------------------------------------------------" + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    print("Job Done!")


def main_3stage(args, train_loader, val_loader, test_loader_CT, test_loader_MM):
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        cudnn.benchmark = True
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    net = create_multimodal_net(args)

    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.base_lr,
                              momentum=args.beta1, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.base_lr,
                              betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    if args.resume:
        net, optimizer, best_mse, start_epoch = load_checkpoint(args, net, optimizer)
    else:
        start_epoch = 0
        best_mse = 1e8
        best_bce_all = 1e8
        best_bce = 1e8
        best_acc1 = 0
        best_acc2 = 0
        best_acc3 = 0
        best_acc4 = 0
        best_sens1 = 0
        best_sens2 = 0
        best_sens3 = 0
        best_sens4 = 0
        stage2_succ = False
        stage3_succ = False

    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Attention Module - " + args.attention_type + "\n")
    for key, val in vars(args).items():
        args.log_file.write("{:16} {}".format(key, val) + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    # multi-GPUs
    if len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, args.gpu_ids)

    if args.CT_report_flag:
        image_model_state_dict = torch.load(args.image_feature_net_para_path)['state_dict']
        feature_extractor_state_dict = {k.replace("feature_extractor.", ""): v
                                        for k, v in image_model_state_dict.items() if
                                        k.startswith("feature_extractor.")}
        net.feature_extractor.load_state_dict(feature_extractor_state_dict)
        for param in net.feature_extractor.parameters():
            param.requires_grad = False
    else:
        for param in net.feature_extractor.parameters():
            param.requires_grad = True
        '''
        for name, module in net.feature_extractor.densenet121.features.named_children():
            if name in ['transition3', 'denseblock4', 'classifier']:
                for param in module.parameters():
                    param.requires_grad = True
        '''

    net.to(args.device)
    summary(net)
    args.log_file.write("Phase 1 training starts.\n")
    if args.classification:
        train_bces = []
        val_bces = []
        test_bces_CT = []
        test_bces_MM = []
    else:
        train_mses = []
        val_mses = []
        test_mses_CT = []
        test_mses_MM = []
    stage = 1 if args.stage3_train else 0
    if args.stage3_train:
        print(f'Training stage{stage} start!')
    stage_epoch = [20, 10, 1000]  # origin: [40, 40, 1000]
    stage_shift = False
    for epoch in range(start_epoch, args.num_epoch):
        if args.reasoning_flag:
            if args.stage3_train:
                if epoch >= sum(stage_epoch[:stage]):
                    stage += 1
                    stage_shift = True
                    best_bce = 1e8
                adjust_learning_rate_mm_3stage(optimizer, epoch, args.base_lr, args.warmup, stage=stage)
            else:
                adjust_learning_rate_mm(optimizer, epoch, args.base_lr, args.warmup)
        else:
            adjust_learning_rate(optimizer, epoch, args.base_lr, args.warmup)

        if args.reasoning_flag:
            if args.stage3_train:
                if stage == 2 and stage_shift:
                    print(f'Training stage{stage} preapare...')
                    print(f'Loading stage{stage - 1} best model...')
                    checkpoint = torch.load(args.ckpt + f'/model_best_checkpoint_stage{stage-1}.pth.tar')
                    # checkpoint = torch.load(args.ckpt + '/model_best_checkpoint.pth.tar')
                    net.load_state_dict(checkpoint['state_dict'], strict=True)
                    net.to(args.device)
                    print('Loading complete.')
                    for param in net.parameters():
                        param.requires_grad = False
                    # Unfreeze all layers about image
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = True

                    for param in net.ln_img.parameters():
                        param.requires_grad = True

                    for param in net.img_self_att.parameters():
                        param.requires_grad = True

                    for param in net.conv1d.parameters():
                        param.requires_grad = True

                    for param in net.mlp_layers[0].parameters():
                        param.requires_grad = True

                    stage_shift = False
                    summary(net)
                    print(f'Training stage{stage} start!')
                if stage == 3 and stage_shift:
                    print(f'Training stage{stage} preapare...')
                    print(f'Loading stage{stage-1} best model...')
                    checkpoint = torch.load(args.ckpt + f'/model_best_checkpoint_stage{stage - 1}.pth.tar')
                    # checkpoint = torch.load(args.ckpt + '/model_best_checkpoint.pth.tar')
                    net.load_state_dict(checkpoint['state_dict'], strict=True)
                    net.to(args.device)
                    print('Loading complete.')
                    for param in net.parameters():
                        param.requires_grad = True
                    stage_shift = False
                    summary(net)
                    print(f'Training stage{stage} start!')
            else:
                if epoch >= stage_epoch[0] and not stage_shift:
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = True
                    stage_shift = True

        print('lr now: %1.8f' % optimizer.param_groups[0]['lr'])
        args.log_file.write('lr now: %1.8f' % optimizer.param_groups[0]['lr'])
        if not os.path.exists(args.ckpt + '/MSE_log/'):
            os.makedirs(args.ckpt + '/MSE_log/')
        if args.classification:
            train_epoch_acc1, train_epoch_sens1, train_epoch_acc2, train_epoch_sens2, train_epoch_acc3, train_epoch_sens3, train_epoch_acc4, train_epoch_sens4, train_epoch_bce = train(net, optimizer, epoch, train_loader, args, stage=stage)
            val_epoch_acc1, val_epoch_sens1, val_epoch_acc2, val_epoch_sens2, val_epoch_acc3, val_epoch_sens3, val_epoch_acc4, val_epoch_sens4, val_epoch_bce = validate(net, epoch, val_loader, args)
            test_CT_epoch_acc1, test_CT_epoch_sens1, test_CT_epoch_acc2, test_CT_epoch_sens2, test_CT_epoch_acc3, test_CT_epoch_sens3, test_CT_epoch_acc4, test_CT_epoch_sens4, test_CT_epoch_bce = test(net, epoch, test_loader_CT, args)
            test_MM_epoch_acc1, test_MM_epoch_sens1, test_MM_epoch_acc2, test_MM_epoch_sens2, test_MM_epoch_acc3, test_MM_epoch_sens3, test_MM_epoch_acc4, test_MM_epoch_sens4, test_MM_epoch_bce = test(net, epoch, test_loader_MM, args)

            train_bces.append(train_epoch_bce)
            val_bces.append(val_epoch_bce)
            test_bces_CT.append(test_CT_epoch_bce)
            test_bces_MM.append(test_MM_epoch_bce)

            np.save(args.ckpt + '/MSE_log/train_bces.npy', np.array(train_bces))
            np.save(args.ckpt + '/MSE_log/val_bces.npy', np.array(val_bces))
            np.save(args.ckpt + '/MSE_log/test_bces_CT.npy', np.array(test_bces_CT))
            np.save(args.ckpt + '/MSE_log/test_bces_MM.npy', np.array(test_bces_MM))

        is_best = val_epoch_bce < best_bce
        is_best_all = val_epoch_bce < best_bce_all
        if is_best_all:
            if stage == 2 and not stage2_succ:
                stage2_succ = True
                print('Stage 2 has an improvement!')
            if stage == 3 and not stage3_succ:
                stage3_succ = True
                print('Stage 3 has an improvement!')
            print('')
        if args.classification:
            best_bce = min(val_epoch_bce, best_bce)
            best_bce_all = min(val_epoch_bce, best_bce_all)
            best_acc1 = max(val_epoch_acc1, best_acc1)
            best_acc2 = max(val_epoch_acc2, best_acc2)
            best_acc3 = max(val_epoch_acc3, best_acc3)
            if args.DLCO_flag:
                best_acc4 = max(val_epoch_acc4, best_acc4)
            best_sens1 = max(val_epoch_sens1, best_sens1)
            best_sens2 = max(val_epoch_sens2, best_sens2)
            best_sens3 = max(val_epoch_sens3, best_sens3)
            if args.DLCO_flag:
                best_sens4 = max(val_epoch_sens4, best_sens4)
            print("best bce: ", best_bce)
            print("best acc 1: ", best_acc1)
            print("best acc 2: ", best_acc2)
            print("best acc 3: ", best_acc3)
            if args.DLCO_flag:
                print("best acc 4: ", best_acc4)
            print("best sens 1: ", best_sens1)
            print("best sens 2: ", best_sens2)
            print("best sens 3: ", best_sens3)
            if args.DLCO_flag:
                print("best sens 4: ", best_sens4)
        # save the best model at the present stage
        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": net.cpu().state_dict(),  # net.module.cpu().state_dict() will error for no data parallel
            "best_mse": best_mse,
            "best_bce": best_bce_all,
            "best_acc1": best_acc1,
            "best_acc2": best_acc2,
            "best_acc3": best_acc3,
            "best_acc4": best_acc4,
            "best_sens1": best_sens1,
            "best_sens2": best_sens2,
            "best_sens3": best_sens3,
            "best_sens4": best_sens4,
            "optimizer": optimizer.state_dict(),
        }, is_best, epoch + 1, save_path=args.ckpt, stage=stage)

        # save the best model in all stages
        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": net.cpu().state_dict(),  # net.module.cpu().state_dict() will error for no data parallel
            "best_mse": best_mse,
            "best_bce": best_bce_all,
            "best_acc1": best_acc1,
            "best_acc2": best_acc2,
            "best_acc3": best_acc3,
            "best_acc4": best_acc4,
            "best_sens1": best_sens1,
            "best_sens2": best_sens2,
            "best_sens3": best_sens3,
            "best_sens4": best_sens4,
            "optimizer": optimizer.state_dict(),
        }, is_best_all, epoch + 1, save_path=args.ckpt, stage=0)

        net.to(args.device)

        args.log_file.write("--------------------------------------------------" + "\n")

    print("Job Done!")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="")

    # Model settings
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="network architecture (default: resnet18)")
    parser.add_argument("--num_base_filters", type=int, default=16,
                        help="network base filer numbers (default: 16)")  # 基础卷积核数量，决定第一层卷积的通道数
    parser.add_argument("--expansion", type=float, default=1,
                        help="expansion factor for the mid-layer in resnet-like")
    parser.add_argument("--block_type", type=str, default="basic",
                        help="building block for network (possible choices basic|bottlenect|ivrd|vgg")
    parser.add_argument("--attention_type", type=str, default="none",
                        help="attention type in building block (possible choices none|se|cbam|wa)")
    parser.add_argument("--attention_param", type=str, default="haar",
                        help="attention parameter (reduction in CBAM and SE, wavename in wavelet)")
    parser.add_argument("--cov_dim", type=int, default=4,
                        help="covariate dimension (default: 4, height, gender, weight, age)")
    parser.add_argument("--res_dim", type=int, default=3,
                        help="response dimension (default: 3, FEV1 VCMAX FVC)")
    parser.add_argument("--image_feature_net_ckpt", type=int, default=None,
                        help="The pre-trained para of image_feature_net for multi-modality model (default: None)")
    parser.add_argument("--CA_num_heads", type=int, default=4,
                        help="The number of heads of Cross Attention Module")
    parser.add_argument("--image_feature_dim", type=int, default=256,
                         help="image feature dimension (default: 64)")
    parser.add_argument("--image_feature_net_para_path", type=str, default='',
                        help="image feature net para save path")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout rate of MLP")
    parser.add_argument("--classification", type=bool, default=False,
                        help="when classification is True, the model outputs class prediction results.")
    parser.add_argument("--percentage_thresholds", type=list, default=[1.0 - 0.18456666, 1.0 - 0.12726749, 1.0 - 0.05931471],
                        help="positive percentage of each task.")

    # Dataset settings
    parser.add_argument("--workers", default=16, type=int,
                        help="number of data loading works")
    parser.add_argument("--ct_image_dir", default=[''], type=list,
                        help="ct image path (default: ./data/lung_mask_20_npy)")
    parser.add_argument("--ct_image_aug", default=False, type=bool,
                        help="when ct_image_aug is true, ct images will be augmented (default: False)")
    parser.add_argument("--pft_data_path", default=[''], type=list,
                        help="pft data path (default: ./data/PFT-merge.xlsx)")
    parser.add_argument("--image_target_shape", default=(256, 256), type=tuple,
                        help="target shape of CT image (default: (256, 256))")
    parser.add_argument("--CT_slice_num", default=20, type=int,
                        help="slice number of CT image (default: 20)")
    parser.add_argument("--CT_report_sentence_num", default=30, type=int,
                        help="max number of CT report sentence (default: 15)")
    parser.add_argument("--reasoning_sentence_num", default=30, type=int,
                        help="max number of CT report sentence (default: 15)")
    parser.add_argument("--CT_report_flag", default=False, type=bool,
                        help="when CT_report_flag is true, CT report is contained (default: True)")
    parser.add_argument("--report_embedding_path",
                        default=['/teams/Thymoma_1685081756/PFT/data/ct_report_30_embedding',
                                 '/teams/Thymoma_1685081756/PFT/data/ct_report_30_embedding_part3'], type=list,
                        help="report embedding path (default: ")
    parser.add_argument("--reasoning_flag", default=False, type=bool,
                        help="when CT_report_flag is true, CT report is contained (default: True)")
    parser.add_argument("--reasoning_embedding_path",
                        default=[''], type=list,
                        help="report embedding path (default: ")
    parser.add_argument("--LLM_predict_path",
                        default=[
                            ''],
                        type=list,
                        help="LLM predict result path")
    parser.add_argument("--LLM_predict_flag", default=False, type=bool,
                        help="LLM_predict_flag (default: False)")
    parser.add_argument("--DLCO_flag", default=True, type=bool,
                        help="When DLCO_flag = True, DLCO is included in responses to be predicted. (default: True)")
    parser.add_argument("--train_val_split", type=float, default=0.2,
                        help="split ratio of train/val dataset (default: 0.2)")
    parser.add_argument("--split_metric", type=str, default='FEV1',
                        help="split metric of dataset (default: 'FEV1')")
    parser.add_argument("--mp_flag", type=bool, default=False,
                        help="when mp_flag is True, the dataset will include the clinically predicted PFT metrics and diagnosis results (default: False)")

    # Optimizion settings
    parser.add_argument("--gpu_ids", default="0",
                        help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size for training and validation (default: 128)")
    parser.add_argument("--num_epoch", type=int, default=200,
                        help="number of epochs to train (default: 200)")
    parser.add_argument("--resume", default="", type=str,
                        help="path to checkpoint for continous training (default: none)")
    parser.add_argument("--optim", default="SGD",
                        help="optimizer")
    parser.add_argument("--base_lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--beta1", default=0.9, type=float,
                        help="momentum for sgd, beta1 for adam")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--warmup", action="store_true",
                        help="warmup for deeper network")
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="test/val dataset split size (default: 0.2)")
    parser.add_argument("--LLM_pos_factor", default=0.5, type=float,
                        help="LLM_pos_factor (default: 0.5)")
    parser.add_argument("--focal_alpha", default=[0.5, 0.5, 0.5], type=list,
                        help="train focal alpha (default: 0.75)")
    parser.add_argument("--val_focal_alpha", default=[0.5, 0.5, 0.5], type=list,
                        help="val focal alpha (default: 0.5)")
    parser.add_argument("--focal_gamma", default=1.0, type=float,
                        help="focal gamma (default: 1.0)")
    parser.add_argument("--stage3_train", default=False, type=bool,
                        help="3 stage train (default: False)")

    # Misc
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--disp_iter", type=int, default=100,
                        help="frequence to display training status (default: 100)")
    parser.add_argument("--ckpt", default="./ckpts/",
                        help="folder to output checkpoints")
    parser.add_argument("--copula_ckpt", default="./copula_ckpts/",
                        help="folder to output copula checkpoints")

    args = parser.parse_args()
    args.gpu_ids = parse_gpus(args.gpu_ids)
    args.classification = True

    args.ct_image_dir = []

    args.CT_slice_num = 10

    args.pft_data_path = []

    args.CT_report_flag = False
    args.CT_report_sentence_num = 1
    args.report_embedding_path = []

    args.reasoning_flag = True
    if args.reasoning_flag:
        args.stage3_train = True
    args.reasoning_embedding_path = []

    args.LLM_predict_flag = True
    args.LLM_factor = [10.0, 10.0, 10.0]
    args.LLM_predict_path = []

    args.DLCO_flag = False

    if not args.DLCO_flag:
        args.reasoning_sentence_num = 3
    else:
        args.reasoning_sentence_num = 4

    args.num_class = 100
    if args.DLCO_flag:
        args.res_dim = 4
    else:
        args.res_dim = 3  # if DLCO is not included args.res_dim = 3, else, args.res_dim = 4
    if args.reasoning_flag:
        if args.DLCO_flag:
            args.cov_dim = 12
        else:
            args.cov_dim = 10
    else:
        args.cov_dim = 4
    args.image_target_shape = (256, 256)
    args.ct_image_aug = True
    args.image_feature_net_para_path = ''
    # basic backbone structure
    # args.arch = "densenet121"
    args.arch = "lightdensenet121"
    args.num_base_filters = 8
    args.block_type = 'bottlenect'
    # attention type
    args.attention_type = 'wa'
    args.attention_param = 'haar'
    # number heads of cross attention
    args.CA_num_heads = 8
    args.image_feature_dim = 128
    args.dropout = 0.2
    args.warmup = False  # When CA is on, warmup is True
    args.optim = 'Adam'
    args.copula_imgNet_finetune = True
    if args.CT_report_flag:
        args.copula_imgNet_finetune = False
    if args.reasoning_flag:
        # train CA module from scratch.
        args.base_lr = 0.001
        args.batch_size = 12
        args.warmup = True
    else:
        # fine tune CheXNet
        args.base_lr = 0.00001
        args.batch_size = 12

    if args.copula_imgNet_finetune:
        args.copula_batch_size = 8
    else:
        args.copula_batch_size = 8
    args.copula_gaussian = False
    args.copula_psi0 = 0.5
    args.num_epoch = 30
    args.focal_alpha = [0.75, 0.75, 0.75]
    args.val_focal_alpha = [0.75, 0.75, 0.75]
    args.focal_gamma = 0.0
    args.split_metric = 'FEV1'
    args.train_val_split = 0.2

    args.ckpt = './ckpts/1022(Ptest-IMGRSFT)/'
    # Phase 1 model save path
    args.ckpt += args.arch
    if args.classification:
        args.ckpt += '-clas'
    else:
        args.ckpt += '-regr'
    args.ckpt += "-res_dim" + str(args.res_dim)
    args.ckpt += "-ctsn" + str(args.CT_slice_num)
    args.ckpt += "-ctr" + str(int(args.CT_report_flag))
    if args.CT_report_flag:
        args.ckpt += "-rsn" + str(args.CT_report_sentence_num)
    args.ckpt += "-reas" + str(int(args.reasoning_flag))
    if args.reasoning_flag:
        args.ckpt += "-rsn" + str(args.reasoning_sentence_num)
    args.ckpt += "-canh" + str(args.CA_num_heads)
    args.ckpt += "-fdim" + str(args.image_feature_dim)
    if args.ct_image_aug:
        args.ckpt += "-imgAug"
    args.ckpt += "-dp" + str(int(args.dropout*100))
    args.ckpt += "-bslr" + str(int(args.base_lr * 1e5))
    if args.stage3_train:
        args.ckpt += "-3ST"
    args.ckpt += "-bs" + str(args.batch_size)
    args.ckpt += "-ssize" + str(args.train_val_split).replace('.', '')
    args.ckpt += "-fa" + '_'.join([str(args.focal_alpha[i]).replace('.', '') for i in range(args.res_dim)])
    args.ckpt += "-fg" + str(args.focal_gamma).replace('.', '')
    args.ckpt += "-llmf_" + '_'.join([str(args.LLM_factor[i]).replace('.', '') for i in range(args.res_dim)])
    if args.val_focal_alpha[0] == 0.5:
        args.ckpt += "-valbce"
    else:
        args.ckpt += "-valcbbce"
    # args.ckpt += "-eqLabel"

    args.seed = 1
    args.ckpt += "-seed" + str(args.seed)
    if args.reasoning_flag:
        args.ckpt += ""
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if not os.path.isdir(args.copula_ckpt):
        os.makedirs(args.copula_ckpt)

    # write to file
    args.log_file = open(os.path.join(args.ckpt, "log_file.txt"), mode="w")
    best_mse1_list = []
    best_mse2_list = []

    seed_everything(args.seed)

    if args.classification:
        args.mp_flag = True
    else:
        args.mp_flag = False
    test_set_names = ['CT', 'P']
    train_val_names = ['YD', 'CJJ', 'MM']
    IDs_train, IDs_val = patient_train_val_split(args.pft_data_path, test_size=args.train_val_split, random_state=args.seed, train_val_marks=train_val_names)

    train_data, val_data, test_data_1, test_data_2, pos_rates = process_image_cov_resp_report(args.ct_image_dir, args.pft_data_path, IDs_train, IDs_val,
                                                         report_flag=args.CT_report_flag,
                                                         report_path=args.report_embedding_path,
                                                         LLM_reasoning=args.reasoning_flag,
                                                         LLM_predict=args.LLM_predict_flag,
                                                         LLM_reasoning_path=args.reasoning_embedding_path,
                                                         LLM_predict_path=args.LLM_predict_path,
                                                         log_file_path='data_process_logfile.txt', mp_flag=args.mp_flag, test_set_names=test_set_names)
    args.percentage_thresholds = [1.0 - pos_rate for pos_rate in pos_rates]
    train_dataloader, val_dataloader, test_dataloader_1, test_dataloader_2 = train_val_dataloader(train_data, val_data, test_data_1, test_data_2, random_state=args.seed, batch_size=args.batch_size, train_data_augment=args.ct_image_aug)

    print('responses_mean :', train_data["responses_mean"])
    print('responses_scale :', train_data["responses_scale"])

    main(args, train_dataloader, val_dataloader, test_dataloader_1, test_dataloader_2)

    args.log_file.close()

    # subprocess.run([sys.executable, "predict_class_img1.py"])

    os.system(
        "export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")
