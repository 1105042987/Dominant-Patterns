import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass

class MSELoss(nn.MSELoss):
    pass

class KLDivLoss(_Loss):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self,pert,dp):
        return F.kl_div(pert.softmax(dim=-1).log(), dp.softmax(dim=-1).repeat(len(pert),1), reduction='batchmean')


class CoSLoss(_WeightedLoss):
    def __init__(self):
        super(CoSLoss, self).__init__()
        self.name='CoS'

    def forward(self, logit_i_p, logit_p, target=None):
        if target is not None:  # label_dependent (deprecated)
            target_logits = (target * logit_i_p).sum(1)
            loss = - 0.05*target_logits - torch.cosine_similarity(logit_p,logit_i_p)
        else:                   # label_free
            loss = 1-torch.cosine_similarity(logit_p, logit_i_p)
        return torch.mean(loss)
