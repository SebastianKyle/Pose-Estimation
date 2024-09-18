import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCCLoss(nn.Module):
    def __init__(
        self, 
        beta=1.0,
        label_beta=10.0,
        label_softmax=False,
        use_target_weight=True, 
        # mask=None,
        # mask_weight=1.0
    ):
        super(SimCCLoss, self).__init__()
        self.beta = beta
        self.label_beta = label_beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, pred, target):
        log_pt = self.log_softmax(pred * self.beta) 
        if self.label_softmax:
            target = F.softmax(target * self.label_beta, dim=1)
        
        loss = torch.mean(self.kl_loss(log_pt, target), dim=1)

        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):

        N, K, _ = pred_simcc[0].shape
        loss = 0
        
        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1

        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            # ? Do we need mask weights
            loss = loss + self.criterion(pred, target).mul(weight).sum()
        
        return loss / K