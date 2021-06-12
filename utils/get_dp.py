import torch,cv2
import sys,os,json5
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
from glob import glob
from tqdm import tqdm
import torchvision.models as models
from dataset.general_image_datasets import dataloader
from docker.tool import yellow

dev = 'cpu'
std = torch.Tensor((0.229, 0.224, 0.225)).reshape(-1,1,1).float().to(dev)
mean = torch.Tensor((0.485, 0.456, 0.406)).reshape(-1,1,1).float().to(dev)
softmax = torch.nn.Softmax(-1).to(dev)
net_dic = {}

with open('RootPath.json') as f:
    RF = json5.load(f)
    root = RF[r'%RESULT%']
    data_root = RF[r'%DATA%']
with open(os.path.join(data_root,'imagenet_class_index.json')) as f:
    num2idx = json5.load(f)


def draw(img_id,img,basepath,target_net,prefix='dp'):
    if target_net not in net_dic:
        net_dic[target_net] = getattr(models,target_net)(pretrained=True).to(dev)
        net_dic[target_net].eval()
    result = softmax(net_dic[target_net]((img-mean)/std))
    prob,idx = result.max(1)
    if img_id>-1:
        name = f'{prefix}_ID{img_id}_CLS({num2idx[str(int(idx))][1]})_Prob{int(prob*100+0.5)}.jpg'
    else:
        name = f'{prefix}_{num2idx[str(int(idx))][1]}_Prob{int(prob*100+0.5)}.jpg'
    if prefix == 'dp':
        temp = ((img[0]-img[0].min()).cpu().numpy().transpose((1,2,0)))
        temp = temp/temp.max()
    else:
        temp = img[0].cpu().numpy().transpose((1,2,0))[...,::-1]
    cv2.imwrite(os.path.join(basepath,'save', name),(temp*255))
    return name

assert len(sys.argv)>1, 'Project Name Needed!'
if len(sys.argv)==3 and sys.argv[2] == 'img':
    num_img = 1
    cfg={
        "dataset_name": "ImageNet",
        "num_workers": 2,
        "direction": data_root,
        "batch_size": 1,
        "shuffle":False,
        "enhancement":{
            "Resize":[256],
            "CenterCrop":[224],
        }
    }
    data = iter(dataloader(cfg,'test'))

tars = glob(os.path.join(root,sys.argv[1],'*'))
tars.sort()
for path in tqdm(tars,ascii=True,desc='EXP NUM',ncols=130):
    if len(glob(os.path.join(path,'*.txt')))!=0:
        w_path = os.path.join(path,'ckp','best','weight.pth')
        if os.path.exists(w_path):
            try:
                weight = torch.load(w_path, map_location=lambda storage, loc:storage)
                dp = weight['dp'].to(dev)
                with open(glob(os.path.join(path,'code','config','*'))[0]) as f:
                    target_net = json5.load(f)["system"]["net_param"]["target_net"]
            except:
                print('error at',yellow(w_path))
                continue
            name = draw(-1,dp,path,target_net,'dp')
            if len(sys.argv)==3 and sys.argv[2] == 'img':
                for idx in tqdm(range(num_img),ascii=True,desc=path.split('/')[-1],ncols=130):
                    img,label = next(data)
                    img = img.to(dev)
                    draw(idx,dp+img,path,target_net,'per')
                    draw(idx,img,path,target_net,'ori')