import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    def __init__(self, LLM_cov, factor=1.0, DLCO_flag=False, device='cpu'):
        super(WeightedMSE, self).__init__()
        if not DLCO_flag:
            res_num = 3
        else:
            res_num = 4
        self.mapped_weights = torch.zeros(LLM_cov.shape[0], 1, dtype=torch.float32)
        self.mapped_weights = self.mapped_weights.to(device, non_blocking=True)
        for i in range(LLM_cov.shape[0]):
            LLM_pos_index = 0.0
            for j in range(res_num):
                if LLM_cov[i, 2 * j] == 1:
                    LLM_pos_index += LLM_cov[i, 2 * j + 1]
                else:
                    LLM_pos_index += 1.0 - LLM_cov[i, 2 * j + 1]
            self.mapped_weights[i] = (factor * (LLM_pos_index / float(res_num))) + (1.0 - factor)

        self.total_weight = self.mapped_weights.sum()

    def forward(self, y_pred, y_true):
        return (self.mapped_weights * (y_pred - y_true) ** 2).sum() / self.total_weight