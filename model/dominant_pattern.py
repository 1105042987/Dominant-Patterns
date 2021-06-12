import torch 
import os,sys
import numpy as np
import torch.nn as nn
from utils import custom_loss
from collections import OrderedDict
import torchvision.models as models
from docker.abstract_model import weak_evaluate, weak_loss
from dataset.general_image_datasets import dataset_feature

loss_name_map ={
    "cos"   : "CoSLoss",
    "mse"   : "MSELOSS",
    "kld"   : "KLDivLoss",
}

def reg_hook(net,place,net_name,sub_layer):
    def hook(models,inputs,outputs):
        place.append(inputs[0])
    if net_name in ['vgg16','vgg19','alexnet','densenet121']:
        if sub_layer==0:
            return net.classifier.register_forward_hook(hook)
        else:
            return net.classifier[sub_layer].register_forward_hook(hook)
    else:
        return net.fc.register_forward_hook(hook) 
    return None

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

class DP(nn.Module):
    def __init__(self, target_net, img_size, pretrain_set, test_as_saved, epsilon, feature_layer):
        super(DP, self).__init__()
        self.img_size = img_size
        self.epsilon = epsilon
        self.dp = nn.Parameter(torch.zeros(size=(1,*img_size), requires_grad=True))
        self.test_as_saved = test_as_saved
        self.hook = None
        self.hidden = None
        self.get_tarNet(target_net,pretrain_set,feature_layer)       
    
    def get_tarNet(self,target_net,pretrain_set,feature_layer):
        self.info = (target_net,pretrain_set,feature_layer)
        self.tarNet = nn.Sequential(
            Normalize(*dataset_feature[pretrain_set][:2]),
            getattr(models,target_net)(pretrained=pretrain_set=='ImageNet'),
        )
        if pretrain_set != 'ImageNet':
            weight = torch.load(f'./pretrain/{target_net}_{pretrain_set}.pth', map_location=lambda storage, loc:storage)
            if target_net in ['vgg16','vgg19','alexnet','densenet121']:
                self.tarNet[1].classifier[6] = nn.Linear(self.tarNet[1].classifier[6].in_features,dataset_feature[pretrain_set][2])
            else:
                self.tarNet[1].fc = nn.Linear(self.tarNet[1].fc.in_features,dataset_feature[pretrain_set][2])
            self.tarNet.load_state_dict(weight)

        for param in self.tarNet.parameters():
            param.requires_grad = False
        if feature_layer > -1:
            if self.hook is not None:
                self.hook.remove()
            self.hidden = []
            self.hook = reg_hook(self.tarNet[1],self.hidden,target_net,feature_layer)

    def state_dict(self):
        return OrderedDict({'dp':self.dp.data,'tar':self.info})
    
    def load_state_dict(self, state_dict):
        if self.test_as_saved:
            assert state_dict.get('tar',None) is not None, "No \"target_net\" saved"
            self.get_tarNet(*state_dict['tar'])
        self.dp.data = torch.Tensor(state_dict['dp'])

    def forward(self, xs):
        self.tarNet.eval()
        self.dp.data = torch.clamp(self.dp.data, -self.epsilon, self.epsilon)
        if self.hidden is not None:
            self.hidden.clear()
        return self.tarNet(xs+self.dp), self.tarNet(xs), self.tarNet(self.dp), self.hidden

class loss(weak_loss):
    def __init__(self, loss_type):
        super(loss, self).__init__()
        self.loss_type = loss_type
        self.loss = getattr(custom_loss,loss_name_map[loss_type])()

    def get_loss(self, pre, tar):
        if pre[3] is not None:
            pre = pre[3]
        loss = self.loss(pre[2],pre[0])
        return loss, {self.loss_type:loss}

class evaluate(weak_evaluate):
    def __init__(self,topks):
        super(evaluate, self).__init__()
        self.topks = topks if type(topks) in (list,tuple) else [topks]
        self.ori_distribution = 0
        self.pert_distribution = 0
        self.dp_class = None

    def acc_calc(self,pre,label,name):
        acc_map = pre.topk(max(self.topks),-1,True,True)[1]==label.reshape(-1,1)
        return {name+f'_top{k}' : acc_map[:,:k].sum(1).float().mean() for k in self.topks}

    def get_eval(self, inputs, preds, targets):
        tar_label = torch.where(targets==1)[1]
        pert, ori, dp = preds[:3]
        ## fooling ratio
        ori_label = ori.argmax(-1)
        pert_label = pert.argmax(-1)
        fool = pert_label != ori_label
        ## accuracy metric
        succ_mask = ori_label==tar_label
        ori_acc = self.acc_calc(ori, tar_label, 'ori_acc')
        pert_acc = self.acc_calc(pert,tar_label,'pert_acc')
        atk_succ = self.acc_calc(pert[succ_mask],tar_label[succ_mask],'atk_succ')
        for k,v in atk_succ.items(): atk_succ[k]=1-v
        ## targeted metric
        self.ori_distribution += (ori == ori.max(1)[0].reshape(-1, 1)).sum(0)
        self.pert_distribution += (pert==pert.max(1)[0].reshape(-1,1)).sum(0)
        self.dp_class = dp.argmax(-1).squeeze()
        return {'fooling_ratio':fool,**ori_acc,**pert_acc,**atk_succ}

    def final_call(self):
        total_num = self.ori_distribution.sum()
        ori_top = self.ori_distribution.topk(max(self.topks),-1,True,True)
        pert_top = self.pert_distribution.topk(max(self.topks), -1, True, True)
        out_dic = {}
        for k in self.topks:
            res = self.dp_class in pert_top[1][:k]
            out_dic['DP_in_top{}'.format(k)] = int(res)
            out_dic['ori_occupy(top{})%'.format(k)] = float(100.0*ori_top[0][:k].sum()/total_num)
            out_dic['pert_occupy(top{})%'.format(k)] = float(100.0*pert_top[0][:k].sum()/total_num)

        return out_dic

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass
