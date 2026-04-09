import torch
import torch.nn as nn


class FocalBCE(nn.Module):
    def __init__(self, diag, alpha=[0.5, 0.5, 0.5], gamma=2.0, DLCO_flag=False, device='cpu'):
        super(FocalBCE, self).__init__()
        if not DLCO_flag:
            self.res_num = 3
        else:
            self.res_num = 4
        self.alpha = alpha
        self.gamma = gamma
        self.mapped_weights = torch.zeros(diag.shape[0], self.res_num, dtype=torch.float32)
        self.mapped_weights = self.mapped_weights.to(device, non_blocking=True)
        self.total_weight = torch.zeros(self.res_num, dtype=torch.float32)
        self.total_weight = self.total_weight.to(device, non_blocking=True)
        for j in range(self.res_num):
            for i in range(diag.shape[0]):
                true_label_now = diag[i, j]
                pos_neg_cc = self.alpha[j] if true_label_now == 1 else (1.0 - self.alpha[j])
                self.mapped_weights[i, j] = pos_neg_cc
            self.total_weight[j] = self.mapped_weights[:, j].sum()
        self.weighted_loss = torch.zeros(self.res_num, dtype=torch.float32)
        self.weighted_loss = self.weighted_loss.to(device, non_blocking=True)

    def forward(self, y_pred, y_true):
        # y_pred is already sigmoid output (probabilities), shape: [N, C] or [N, 1]
        # y_true shape must match y_pred
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        assert y_true.shape[1] == self.res_num, f"Expected {self.res_num} outputs, but got {y_true.shape[1]}"
        eps = 1e-8  # to avoid log(0)
        for j in range(self.res_num):
            # bce_loss = - (y_true[:, j] * torch.log(y_pred[:, j] + eps) + (1 - y_true[:, j]) * torch.log(1 - y_pred[:, j] + eps))
            bce_loss = - (y_true[:, j] * (1.0 - y_pred[:, j])**self.gamma * torch.log(y_pred[:, j] + eps) + (1 - y_true[:, j]) * y_pred[:, j]**self.gamma * torch.log(
                1 - y_pred[:, j] + eps))
            self.weighted_loss[j] = (self.mapped_weights[:, j] * bce_loss).sum() / self.total_weight[j]
        return self.weighted_loss.sum() / self.res_num


class WeightedBCE(nn.Module):
    def __init__(self, LLM_cov, diag, factor=[1.0, 1.0, 1.0], DLCO_flag=False, device='cpu', alpha=[0.5, 0.5, 0.5], gamma=2.0, fac_low_cc=True):
        super(WeightedBCE, self).__init__()
        if not DLCO_flag:
            self.res_num = 3
        else:
            self.res_num = 4
        self.alpha = alpha
        self.gamma = gamma
        self.mapped_weights = torch.zeros(LLM_cov.shape[0], self.res_num, dtype=torch.float32)
        self.mapped_weights = self.mapped_weights.to(device, non_blocking=True)
        self.total_weight = torch.zeros(self.res_num, dtype=torch.float32)
        self.total_weight = self.total_weight.to(device, non_blocking=True)
        for j in range(self.res_num):
            for i in range(LLM_cov.shape[0]):
                LLM_uc_index = LLM_cov[i, 2 * j + 1]
                true_label_now = diag[i, j]
                pos_neg_cc = self.alpha[j] if true_label_now == 1 else (1.0 - self.alpha[j])
                if LLM_uc_index > 0.75:
                    self.mapped_weights[i, j] = factor[j] if not fac_low_cc else 1.0
                    self.mapped_weights[i, j] = self.mapped_weights[i, j] * pos_neg_cc
                else:
                    self.mapped_weights[i, j] = factor[j] if fac_low_cc else 1.0
                    self.mapped_weights[i, j] = self.mapped_weights[i, j] * pos_neg_cc
                # self.mapped_weights[i, j] = (factor[j] * LLM_uc_index) + (1.0 - factor[j])
            self.total_weight[j] = self.mapped_weights[:, j].sum()
        self.weighted_loss = torch.zeros(self.res_num, dtype=torch.float32)
        self.weighted_loss = self.weighted_loss.to(device, non_blocking=True)

    def forward(self, y_pred, y_true):
        # y_pred is already sigmoid output (probabilities), shape: [N, C] or [N, 1]
        # y_true shape must match y_pred
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        assert y_true.shape[1] == self.res_num, f"Expected {self.res_num} outputs, but got {y_true.shape[1]}"
        eps = 1e-8  # to avoid log(0)
        for j in range(self.res_num):
            bce_loss = - (y_true[:, j] * (1.0 - y_pred[:, j])**self.gamma * torch.log(y_pred[:, j] + eps) + (1 - y_true[:, j]) * y_pred[:, j]**self.gamma * torch.log(1 - y_pred[:, j] + eps))
            self.weighted_loss[j] = (self.mapped_weights[:, j] * bce_loss).sum() / self.total_weight[j]
        return self.weighted_loss.sum() / self.res_num