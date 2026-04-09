from __future__ import absolute_import

# system lib
import os
import time
import sys
import argparse
# numerical libs
import random
from utils.util import seed_everything
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchinfo import summary
# model_ckpts
# from thop import profile
from utils.util import AverageMeter, ProgressMeter, accuracy, parse_gpus, meanSquaredError
from utils.checkpoint import save_checkpoint, load_checkpoint
from models.multimodal import create_net, create_multimodal_net
from models.CopulaLoss import CopulaLoss
from models.CopulaLoss import copula_loss_instantiation
from Data import CustomDataset, process_image_cov_resp_report, train_val_dataloader, patient_train_val_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns


def find_best_threshold(y_true, y_prob, thresholds, mode='bl'):
    best_thresh = None
    best_score = -1e8 if mode in ['yd', 'f2', 'f15', 'f1'] else 1e8

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        if mode == 'yd':
            score = sensitivity + specificity - 1
            if score > best_score:
                best_score = score
                best_thresh = thresh

        elif mode == 'bl':
            score = abs(sensitivity - specificity)
            if score < best_score:
                best_score = score
                best_thresh = thresh

        elif mode == 'f2':
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision + recall == 0:
                f2 = 0
            else:
                f2 = (5 * precision * recall) / (4 * precision + recall)
            if f2 > best_score:
                best_score = f2
                best_thresh = thresh
        elif mode == 'f15':
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision + recall == 0:
                f15 = 0
            else:
                f15 = (3.25 * precision * recall) / (2.25 * precision + recall)
            if f15 > best_score:
                best_score = f15
                best_thresh = thresh

        elif mode == 'f1':
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (2.0 * precision * recall) / (precision + recall)
            if f1 > best_score:
                best_score = f1
                best_thresh = thresh

        else:
            raise ValueError("mode must be one of ['yd', 'bl', 'f2', 'f15', 'f1']")

    return best_thresh


def bootstrap_best_threshold(y_true, y_prob, thresholds_roc, n_bootstrap=100, mode='bl'):
    # thresholds = np.linspace(0, 1, 1000)
    thresholds = thresholds_roc
    best_thresholds = []
    if n_bootstrap > 1:
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_true_sample = y_true[indices]
            y_prob_sample = y_prob[indices]
            best_thresh = find_best_threshold(y_true_sample, y_prob_sample, thresholds, mode=mode)
            best_thresholds.append(best_thresh)
    else:
        best_thresh = find_best_threshold(y_true, y_prob, thresholds, mode=mode)
        best_thresholds.append(best_thresh)
    # return np.median(best_thresholds), best_thresholds
    return np.mean(best_thresholds), best_thresholds


def plot_roc_curves(predicted_prob, true_label, save_path='', bestt=False, title_name='', n_bootstrap=50, save_fig=False, save_data=False):

    plt.figure(figsize=(predicted_prob.shape[1] * 5, 5))
    optimal_thresholds_yd = []
    optimal_thresholds_f2 = []
    optimal_thresholds_f15 = []
    optimal_thresholds_f1 = []
    optimal_thresholds_bl = []
    cand_thresholds = np.linspace(0, 1, 1000).tolist()
    for i in range(predicted_prob.shape[1]):
        fpr, tpr, thresholds_roc = roc_curve(true_label[:, i], predicted_prob[:, i])
        # print("len(thresholds_roc): ", len(thresholds_roc))
        # print(thresholds_roc)
        roc_auc = auc(fpr, tpr)
        if save_data:
            np.save(save_path + f'{title_name}_task{i}_fprs.npy', np.array(fpr))
            np.save(save_path + f'{title_name}_task{i}_tprs.npy', np.array(tpr))
            np.save(save_path + f'{title_name}_task{i}_roc_auc.npy', np.array(roc_auc))

        if bestt:
            best_thresh_yd, _ = bootstrap_best_threshold(true_label[:, i], predicted_prob[:, i], thresholds_roc, mode='yd', n_bootstrap=n_bootstrap)
            best_thresh_f2, _ = bootstrap_best_threshold(true_label[:, i], predicted_prob[:, i], thresholds_roc, mode='f2', n_bootstrap=n_bootstrap)
            best_thresh_f15, _ = bootstrap_best_threshold(true_label[:, i], predicted_prob[:, i], thresholds_roc, mode='f15', n_bootstrap=n_bootstrap)
            best_thresh_f1, _ = bootstrap_best_threshold(true_label[:, i], predicted_prob[:, i], thresholds_roc, mode='f1', n_bootstrap=n_bootstrap)
            best_thresh_bl, _ = bootstrap_best_threshold(true_label[:, i], predicted_prob[:, i], thresholds_roc, mode='bl', n_bootstrap=n_bootstrap)
            optimal_thresholds_yd.append(best_thresh_yd)
            optimal_thresholds_f2.append(best_thresh_f2)
            optimal_thresholds_f15.append(best_thresh_f15)
            optimal_thresholds_f1.append(best_thresh_f1)
            optimal_thresholds_bl.append(best_thresh_bl)

        plt.subplot(1, predicted_prob.shape[1], i + 1)

        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        # if bestt:
        #     plt.scatter(fpr[idx], tpr[idx], marker='o', color='red', s=80,
        #                 label=f'Optimal {best_thre_mode} Threshold (J={J[idx]:.2f})\nThreshold={optimal_threshold:.6f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_name} ROC Curve for Task {i + 1}')
        plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if not os.path.exists(save_path + 'ROC_curves/'):
        os.mkdir(save_path + 'ROC_curves/')
    if save_fig:
        plt.savefig(save_path + 'ROC_curves/' + f'{title_name}_ROC_Curves.png', dpi=600)
    # plt.show()
    plt.close()
    if bestt:
        for i, thresh in enumerate(optimal_thresholds_yd):
            print(f"Task {i + 1} Optimal Threshold (youden Index): {thresh:.8f}")
        for i, thresh in enumerate(optimal_thresholds_f2):
            print(f"Task {i + 1} Optimal Threshold (f2 Index): {thresh:.8f}")
        for i, thresh in enumerate(optimal_thresholds_f15):
            print(f"Task {i + 1} Optimal Threshold (f15 Index): {thresh:.8f}")
        for i, thresh in enumerate(optimal_thresholds_f1):
            print(f"Task {i + 1} Optimal Threshold (f1 Index): {thresh:.8f}")
        for i, thresh in enumerate(optimal_thresholds_bl):
            print(f"Task {i + 1} Optimal Threshold (balance Index): {thresh:.8f}")
        if save_data:
            np.save(save_path + f'{title_name}_yd_best_thres.npy', np.array(optimal_thresholds_yd))
            np.save(save_path + f'{title_name}_f2_best_thres.npy', np.array(optimal_thresholds_f2))
            np.save(save_path + f'{title_name}_f15_best_thres.npy', np.array(optimal_thresholds_f15))
            np.save(save_path + f'{title_name}_f1_best_thres.npy', np.array(optimal_thresholds_f1))
            np.save(save_path + f'{title_name}_bl_best_thres.npy', np.array(optimal_thresholds_bl))
    return optimal_thresholds_yd, optimal_thresholds_f2, optimal_thresholds_f15, optimal_thresholds_f1, optimal_thresholds_bl


def calculate_metrics(y_true, y_prob, threshold, save_path='', title_name='', task_id=1, save_fig=False):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'{title_name} Confusion Matrix Task{task_id}', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(save_path + 'confusion_matrix/'):
        os.mkdir(save_path + 'confusion_matrix/')
    if save_fig:
        plt.savefig(save_path + 'confusion_matrix/' + f'{title_name}_confusion_matrix_Task{task_id}.png', dpi=600)
    # plt.show()
    plt.close()
    return accuracy, npv, ppv, specificity, sensitivity


def calculate_metrics_bootstrap(y_true, y_prob, threshold, B=100, ci=0.95):
    n = len(y_true)
    metrics_list = {'accuracy': [], 'npv': [], 'ppv': [], 'specificity': [], 'sensitivity': [], 'auc': []}

    for b in range(B):
        idx = np.random.choice(n, n, replace=True)
        y_true_b = y_true[idx]
        y_prob_b = y_prob[idx]
        y_pred_b = (y_prob_b >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        try:
            fpr, tpr, _ = roc_curve(y_true_b, y_prob_b)
            auc_value = auc(fpr, tpr)
        except Exception:
            auc_value = np.nan

        metrics_list['accuracy'].append(accuracy)
        metrics_list['npv'].append(npv)
        metrics_list['ppv'].append(ppv)
        metrics_list['specificity'].append(specificity)
        metrics_list['sensitivity'].append(sensitivity)
        metrics_list['auc'].append(auc_value)

    def summary_ci(values):
        values = np.array(values)
        median = np.nanmedian(values)
        lower = np.nanpercentile(values, (1 - ci) / 2 * 100)
        upper = np.nanpercentile(values, (1 + ci) / 2 * 100)
        return [median, lower, upper]

    accuracy_ci = summary_ci(metrics_list['accuracy'])
    npv_ci = summary_ci(metrics_list['npv'])
    ppv_ci = summary_ci(metrics_list['ppv'])
    specificity_ci = summary_ci(metrics_list['specificity'])
    sensitivity_ci = summary_ci(metrics_list['sensitivity'])
    auc_ci = summary_ci(metrics_list['auc'])
    return accuracy_ci, npv_ci, ppv_ci, specificity_ci, sensitivity_ci, auc_ci, metrics_list


def plot_dca(y_prob, y_true, save_path='', title_name='', task_id=1, save_fig=False):
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()
    N = len(y_true)
    prevalence = np.mean(y_true)

    thresholds = np.linspace(0.0001, 0.9999, 10000)
    net_benefit_model = []
    net_benefit_all = []

    for pt in thresholds:
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb = (float(tp) / float(N) - float(fp) / float(N) * (pt / (1 - pt))) / prevalence
        net_benefit_model.append(nb)

        nb_all = (prevalence - (1 - prevalence) * (pt / (1 - pt))) / prevalence
        net_benefit_all.append(nb_all)

    net_benefit_none = np.zeros_like(thresholds)

    net_benefit_model = np.maximum(net_benefit_model, 0)
    net_benefit_model[-1500:] = 0

    # Plot
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, net_benefit_none, label='Treat None', color='black', zorder=1)
    plt.plot(thresholds, net_benefit_all, label='Treat All', linestyle='--', color='gray', zorder=2)
    plt.plot(thresholds, net_benefit_model, label='Model', color='#FF69B4', zorder=3)

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f'{title_name}_DCA_Curve_Task{task_id}')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1)
    plt.tight_layout()
    if not os.path.exists(save_path + 'DCA_curves/'):
        os.mkdir(save_path + 'DCA_curves/')
    if save_fig:
        plt.savefig(save_path + 'DCA_curves/' + f'{title_name}_DCA_Curve_Task{task_id}.png', dpi=600)
    # plt.show()
    plt.close()


def calculate_all_metrics(predicted_prob, true_label, best_thresholds, save_path='', title_name='', save_fig=False):
    results = []
    for i in range(predicted_prob.shape[1]):
        acc, npv, ppv, spec, sens = calculate_metrics(true_label[:, i], predicted_prob[:, i], best_thresholds[i], save_path=save_path, title_name=title_name, task_id=i+1, save_fig=save_fig)
        results.append({
            'Task': f'Task_{i+1}',
            'Accuracy': acc,
            'NPV': npv,
            'PPV': ppv,
            'Specificity': spec,
            'Sensitivity': sens,
            'best_threshold': best_thresholds[i]
        })
    return results


def calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds):
    results = []
    metrics_boots_values_list = []
    for i in range(predicted_prob.shape[1]):
        acc_ci, npv_ci, ppv_ci, spec_ci, sens_ci, auc_ci, metrics_list = calculate_metrics_bootstrap(true_label[:, i], predicted_prob[:, i], best_thresholds[i])
        results.append({
            'Task': f'Task_{i+1}',
            'Accuracy_median': acc_ci[0],
            'Accuracy_2.5': acc_ci[1],
            'Accuracy_97.5': acc_ci[2],
            'NPV_median': npv_ci[0],
            'NPV_2.5': npv_ci[1],
            'NPV_97.5': npv_ci[2],
            'PPV_median': ppv_ci[0],
            'PPV_2.5': ppv_ci[1],
            'PPV_97.5': ppv_ci[2],
            'Specificity_median': spec_ci[0],
            'Specificity_2.5': spec_ci[1],
            'Specificity_97.5': spec_ci[2],
            'Sensitivity_median': sens_ci[0],
            'Sensitivity_2.5': sens_ci[1],
            'Sensitivity_97.5': sens_ci[2],
            'AUC_median': auc_ci[0],
            'AUC_2.5': auc_ci[1],
            'AUC_97.5': auc_ci[2],
            'best_threshold': best_thresholds[i]
        })
        metrics_boots_values_list.append(metrics_list)
    return results, metrics_boots_values_list


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
    parser.add_argument("--percentage_thresholds", type=list,
                        default=[1.0 - 0.18456666, 1.0 - 0.12726749, 1.0 - 0.05931471],
                        help="positive percentage of each task.")

    # Dataset settings
    parser.add_argument("--workers", default=16, type=int,
                        help="number of data loading works")
    parser.add_argument("--ct_image_dir", default=[''],
                        type=list,
                        help="ct image path (default: ./data/lung_mask_20_npy)")
    parser.add_argument("--ct_image_aug", default=False, type=bool,
                        help="when ct_image_aug is true, ct images will be augmented (default: False)")
    parser.add_argument("--pft_data_path",
                        default=[''], type=list,
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
                        default=[''], type=list,
                        help="report embedding path (default: ")
    parser.add_argument("--reasoning_flag", default=False, type=bool,
                        help="when CT_report_flag is true, CT report is contained (default: True)")
    parser.add_argument("--reasoning_embedding_path",
                        default=[
                            ''],
                        type=list,
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
    args.reasoning_embedding_path = []

    args.LLM_predict_flag = True
    args.LLM_factor = [1.0, 1.0, 1.0]
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
    args.ct_image_aug = False
    args.image_feature_net_para_path = ''
    # basic backbone structure
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
    args.num_epoch = 50
    args.focal_alpha = [0.8, 0.9, 0.9]
    args.val_focal_alpha = [0.8, 0.9, 0.9]
    args.focal_gamma = 0.0
    args.split_metric = 'FEV1'
    args.train_val_split = 0.2
    if args.classification:
        args.mp_flag = True
    else:
        args.mp_flag = False

    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        cudnn.benchmark = True
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    net = create_multimodal_net(args)
    summary(net)
    model_name = 'IMG-RS'
    model_path = ''
    result_root_path = f'ckpts/results/{model_name}/'
    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)

    args.seed = 1

    seed_everything(args.seed)

    test_set_names = ['CT', 'YX']
    train_val_names = ['YD', 'CJJ', 'MM']
    IDs_train, IDs_val = patient_train_val_split(args.pft_data_path, test_size=args.train_val_split, random_state=args.seed, train_val_marks=train_val_names)
    # IDs_train = IDs_train[:100]
    train_data, val_data, test_data_1, test_data_2, _ = process_image_cov_resp_report(args.ct_image_dir,
                                                                                         args.pft_data_path, IDs_train,
                                                                                         IDs_val,
                                                                                         report_flag=args.CT_report_flag,
                                                                                         report_path=args.report_embedding_path,
                                                                                         LLM_reasoning=args.reasoning_flag,
                                                                                         LLM_predict=args.LLM_predict_flag,
                                                                                         LLM_reasoning_path=args.reasoning_embedding_path,
                                                                                         LLM_predict_path=args.LLM_predict_path,
                                                                                         log_file_path='data_process_logfile.txt',
                                                                                         mp_flag=args.mp_flag, test_set_names=test_set_names)
    train_dataloader, val_dataloader, test_dataloader_1, test_dataloader_2 = \
        train_val_dataloader(train_data, val_data, test_data_1, test_data_2, random_state=args.seed,
                             batch_size=args.batch_size, train_data_augment=args.ct_image_aug,
                             train_dataset_shuffle=False)

    data_loader_list = [val_dataloader, train_dataloader, test_dataloader_1, test_dataloader_2]
    data_list = [val_data, train_data, test_data_1, test_data_2]
    save_fig = True
    save_mid_data = True

    # 20260319 update
    pft0216_path = args.pft_data_path[-1]
    df_pft0216 = pd.read_excel(pft0216_path).copy()
    for l in range(3):
        checkpoint = torch.load(model_path + f'model_best_checkpoint_stage{l+1}.pth.tar')
        checkpoint_name = f'ckpt_best_stage{l+1}'
        if not os.path.exists(model_path + 'result/'):
            os.mkdir(model_path + 'result/')
        # result_save_path = model_path + 'result/' + checkpoint_name + '/'
        result_save_path = result_root_path + checkpoint_name + '/'
        if not os.path.exists(result_save_path):
            os.mkdir(result_save_path)
        beta = 1.0
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        net.to(args.device)
        filename_list = ['val', 'train', f'test_{test_set_names[0]}', f'test_{test_set_names[1]}_v5']
        filename_list = ['DLCO' + str(int(args.DLCO_flag)) + '_' + checkpoint_name + '_' + fn for fn in filename_list]
        best_thresholds = []
        n_bootstrap = 1
        for i, data_loader in enumerate(data_loader_list):
            data = data_list[i]
            filename = filename_list[i]
            # 20260216 update
            if 'test_YX' not in filename and 'val' not in filename:
                continue
            print(f"filename now : {filename}")
            predicted_prob = []
            true_label = []
            net.eval()
            with torch.no_grad():
                for batch_idx, (image, x, report, reasoning, reasoning_mask, target, diag) in enumerate(data_loader):
                    # seed_everything(args.seed)
                    image, x, target, diag = image.to(args.device, non_blocking=True), \
                                             x.to(args.device, non_blocking=True), \
                                             target.to(args.device, non_blocking=True), \
                                             diag.to(args.device, non_blocking=True)

                    if args.CT_report_flag:
                        report = report.to(args.device, non_blocking=True)
                    if args.reasoning_flag:
                        reasoning = reasoning.to(args.device, non_blocking=True)
                        reasoning_mask = reasoning_mask.to(args.device, non_blocking=True)
                    output = net(image, x, report=report, reasoning=reasoning, reasoning_mask=reasoning_mask)
                    predicted_prob.append(output.cpu().numpy())
                    true_label.append(diag.cpu().numpy())
            predicted_prob = np.vstack(predicted_prob).astype(np.float32)
            true_label = np.vstack(true_label).astype(np.int32)

            if i == 0:
                bestt = True
                best_thresholds_yd, best_thresholds_f2, best_thresholds_f15, best_thresholds_f1, best_thresholds_bl = plot_roc_curves(predicted_prob, true_label, save_path=result_save_path, bestt=bestt, title_name=filename_list[i], n_bootstrap=n_bootstrap, save_data=False, save_fig=False)
                # 20260216 update
                if l == 0:
                    best_thresholds_bl = [0.484672009944916, 0.328782796859741, 0.222325384616852]
                elif l == 1:
                    best_thresholds_bl = [0.44843527674675, 0.297327697277069, 0.206014484167099]
                else:
                    best_thresholds_bl = [0.718680202960968, 0.314318090677261, 0.0967199355363846]

            else:
                bestt = False
                plot_roc_curves(predicted_prob, true_label, save_path=result_save_path, bestt=bestt, title_name=filename_list[i], save_data=save_mid_data, save_fig=save_fig)

            for j in range(predicted_prob.shape[1]):
                plot_dca(predicted_prob[:, j], true_label[:, j], save_path=result_save_path, title_name=filename_list[i], task_id=j+1, save_fig=save_fig)

            metrics_yd = calculate_all_metrics(predicted_prob, true_label, best_thresholds_yd, title_name=filename_list[i] + '_yd', save_path=result_save_path, save_fig=save_fig)

            metrics_f2 = calculate_all_metrics(predicted_prob, true_label, best_thresholds_f2, title_name=filename_list[i] + '_f2', save_path=result_save_path, save_fig=save_fig)

            metrics_f15 = calculate_all_metrics(predicted_prob, true_label, best_thresholds_f15, title_name=filename_list[i] + '_f15', save_path=result_save_path, save_fig=save_fig)

            metrics_f1 = calculate_all_metrics(predicted_prob, true_label, best_thresholds_f1, title_name=filename_list[i] + '_f1', save_path=result_save_path, save_fig=save_fig)

            metrics_bl = calculate_all_metrics(predicted_prob, true_label, best_thresholds_bl, title_name=filename_list[i] + '_bl', save_path=result_save_path, save_fig=save_fig)

            metrics_mid = calculate_all_metrics(predicted_prob, true_label, [0.5, 0.5, 0.5], title_name=filename_list[i] + '_mid', save_path=result_save_path, save_fig=save_fig)

            metrics_dict_list = [metrics_yd, metrics_f2, metrics_f15, metrics_f1, metrics_bl, metrics_mid]

            metrics_boot_yd, metrics_boot_data_yd = calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds_yd)

            metrics_boot_f2, metrics_boot_data_f2 = calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds_f2)

            metrics_boot_f15, metrics_boot_data_f15 = calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds_f15)

            metrics_boot_f1, metrics_boot_data_f1 = calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds_f1)

            metrics_boot_bl, metrics_boot_data_bl = calculate_all_metrics_bootstrap(predicted_prob, true_label, best_thresholds_bl)

            metrics_boot_mid, metrics_boot_data_mid = calculate_all_metrics_bootstrap(predicted_prob, true_label, [0.5, 0.5, 0.5])

            metrics_boot_dict_list = [metrics_boot_yd, metrics_boot_f2, metrics_boot_f15, metrics_boot_f1,
                                      metrics_boot_bl, metrics_boot_mid]
            metrics_boot_data_list = [metrics_boot_data_yd, metrics_boot_data_f2, metrics_boot_data_f15, metrics_boot_data_f1,
                                      metrics_boot_data_bl, metrics_boot_data_mid]

            index_name_list = ['yd', 'f2', 'f15', 'f1', 'bl', 'mid']
            print("----------------------------------------------------------------------")

            for k, metrics in enumerate(metrics_dict_list):
                print("\n")
                print("Optimal Index", index_name_list[k])
                print("\n")
                for m in metrics:
                    print(filename_list[i] + ' set: ')
                    print(
                        f"Task {m['Task']}: Accuracy={m['Accuracy']:.3f}, NPV={m['NPV']:.3f}, PPV={m['PPV']:.3f}, Specificity={m['Specificity']:.3f}, Sensitivity={m['Sensitivity']:.3f}")

                mdf = pd.DataFrame(metrics)
                if not os.path.exists(result_save_path + 'metrics/'):
                    os.mkdir(result_save_path + 'metrics/')
                mdf.to_excel(result_save_path + 'metrics/' + f'{index_name_list[k]}_{filename_list[i]}_class_metrics_results.xlsx', index=False)
            for k, metrics_boot in enumerate(metrics_boot_dict_list):
                mdf = pd.DataFrame(metrics_boot)
                if not os.path.exists(result_save_path + 'metrics_boot/'):
                    os.mkdir(result_save_path + 'metrics_boot/')
                mdf.to_excel(
                    result_save_path + 'metrics_boot/' + f'{index_name_list[k]}_{filename_list[i]}_class_metrics_boot_results.xlsx',
                    index=False)

            for k, metrics_boot_datas in enumerate(metrics_boot_data_list):
                for q, metrics_boot_data in enumerate(metrics_boot_datas):
                    boots_df = pd.DataFrame(metrics_boot_data)
                    if not os.path.exists(result_save_path + 'metrics_boot_values/'):
                        os.mkdir(result_save_path + 'metrics_boot_values/')
                    boots_df.to_excel(
                        result_save_path + 'metrics_boot_values/' + f'Task{q}_{index_name_list[k]}_{filename_list[i]}_class_metrics_boot_values.xlsx',
                        index=False)
            print("----------------------------------------------------------------------")
            print("Job Done.")
