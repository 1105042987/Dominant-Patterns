import torch
import os,sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
from os.path import join as PJ
import torchvision.transforms as T
import torchvision.datasets as dset
from dataset.auto_download_coco import auto_download

dataset_feature = {
    'ImageNet':((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),1000),
    'COCO':    ((0.40789654, 0.44719302, 0.47026115), 
                (0.28863828, 0.27408164, 0.27809835),90),
    'CIFAR10': ((0.,0.,0.),(1.,1.,1.),10),
    'BigGAN':   ((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225), 1000),
}

class ToLabel(object):
    def __init__(self):
        pass
    def __call__(self, target):
        labels = torch.Tensor([t['category_id'] for t in target])
        if len(labels)==0: return 0
        return labels.long().bincount().argmax()

class ToOneHot(object):
    def __init__(self, num_cls):
        self.num_cls = num_cls
    def __call__(self, label):
        label = torch.Tensor([label]).long()
        label[label>=self.num_cls]=self.num_cls-1
        return torch.nn.functional.one_hot(label, self.num_cls).squeeze()
    def __repr__(self):
        return self.__class__.__name__ + '()'


def dataloader(cfg, mode):
    base_dir = PJ(cfg['direction'],cfg['dataset_name'])
    transform = []
    for key,val in cfg['enhancement'].items():
        transform.append(getattr(T,key)(*val))
    transform.append(T.ToTensor())
    datatype = 'val' if mode == 'test' else mode
    target_transform = [ToLabel()] if cfg['dataset_name']=='COCO' else []
    target_transform.append(ToOneHot(dataset_feature[cfg['dataset_name']][2]))

    param_dict={
        "transform":T.Compose(transform),
        "target_transform":T.Compose(target_transform)
    }
    if cfg['dataset_name'][:5]=='CIFAR':
        cfg['data_class_name'] = cfg['dataset_name']
        param_dict.update({
            'root': base_dir,
            'train': mode=='train',
            'download': True
        })
    elif cfg['dataset_name']=='COCO':
        cfg['data_class_name'] = 'CocoDetection'
        year = '2017'
        auto_download(base_dir,datatype,year)
        param_dict.update({
            "root": PJ(base_dir,datatype+year),
            "annFile": PJ(base_dir, 'annotations', f'instances_{datatype+year}.json'),
        })
    elif cfg['dataset_name'] == 'ImageNet':
        cfg['data_class_name'] = 'ImageFolder'
        param_dict['root'] = PJ(cfg['direction'], 'ILSVRC2012_img_val', datatype)
    elif cfg['dataset_name'] == 'BigGAN':
        cfg['data_class_name'] = 'ImageFolder'
        param_dict['root'] = PJ(base_dir, datatype)
    print('Load data at {}'.format(param_dict['root']))
    
    dataset = getattr(dset,cfg['data_class_name'])(**param_dict)
    print(f'{mode} data has {len(dataset)} images.')
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers'])
    return loader


if __name__ == "__main__":
    import json5
    with open('RootPath.json') as f:
        RF = json5.load(f)
        data_root = RF[r'%DATA%']
    cfg={
        "dataset_name": "COCO",
        "num_workers": 2,
        "direction": data_root,
        "batch_size": 32,
        "single_image_seed": None,
        "shuffle":True,
        "enhancement":{
            "Resize":[256],
            "RandomCrop":[224],
            "RandomHorizontalFlip":[0.5],
            "RandomVerticalFlip":[0.5]
        }
    }
    data = dataloader(cfg,'train')
    from tqdm import tqdm
    for i,t in tqdm(data):
        print(t)
