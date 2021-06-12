import os,sys
import torch 
import numpy as np
import torch.nn as nn
import torchvision.models as models
from docker.abstract_model import weak_evaluate, weak_loss
from collections import OrderedDict
from dataset.general_image_datasets import dataset_feature

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = nn.Parameter(
            torch.Tensor(mean).reshape(-1,1,1).float(),
            requires_grad=False
        )
        self.std = nn.Parameter(
            torch.Tensor(std).reshape(-1,1,1).float(),
            requires_grad=False
        )
    def forward(self, tensor):
        return (tensor-self.mean)/self.std

def grad_decay_decorator(decay_rate):
    def calc_grad(grad):
        return decay_rate*grad
    return calc_grad

class classifier(nn.Module):
    def __init__(self, target_net, pretrain_set, finetune, grad_decay_rate):
        super(classifier, self).__init__()
        self.decay_rate = grad_decay_rate
        self.get_tarNet(target_net,pretrain_set,finetune)
    
    def get_tarNet(self,target_net,pretrain_set,finetune):
        self.info = (target_net,pretrain_set,finetune)
        self.tarNet = nn.Sequential(
            Normalize(*dataset_feature[pretrain_set][:2]),
            getattr(models,target_net)(pretrained=True),
        )
        for mod in list(self.tarNet[1].children())[:-1]:
            for param in mod.parameters():
                if self.decay_rate==0:
                    param.requires_grad = False
                else:
                    param.register_hook(grad_decay_decorator(self.decay_rate))
        if target_net in ['vgg16','vgg19','alexnet','densenet121']:
            self.tarNet[1].classifier[6] = nn.Linear(self.tarNet[1].classifier[6].in_features,dataset_feature[finetune][2])
        else:
            self.tarNet[1].fc = nn.Linear(self.tarNet[1].fc.in_features,dataset_feature[finetune][2])
    
    def load_state_dict(self,dic):
        self.tarNet.load_state_dict(dic)

    def state_dict(self):
        return self.tarNet.state_dict()

    def forward(self, xs):
        return self.tarNet(xs)


class loss(weak_loss):
    def __init__(self):
        super(loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, pre, tar):
        tar = torch.where(tar==1)[1]
        loss=self.loss(pre,tar)
        return loss, {'ce':loss}

class evaluate(weak_evaluate):
    def __init__(self,topks):
        super(evaluate, self).__init__()
        self.topks = topks if type(topks) in (list,tuple) else [topks]

    def acc_calc(self,pre,label,name):
        acc_map = pre.topk(max(self.topks),-1,True,True)[1]==label.reshape(-1,1)
        return {name+'_top{}'.format(k) : acc_map[:,:k].sum(1).float().mean() for k in self.topks}

    def get_eval(self, inputs, preds, targets):
        tar_label = torch.where(targets==1)[1]
        ori_acc = self.acc_calc(preds, tar_label, 'ori_acc')
        return ori_acc

    def final_call(self):
        return {}

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass
        
