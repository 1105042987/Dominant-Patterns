import shutil
import os,sys
import json,json5
import argparse
import traceback
import numpy as np
from pprint import pprint
from datetime import datetime
from docker.tool import yellow

def base_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument('-epoch', nargs='?', type=str, default=None,
                        help="Start epoch, Default -1 (Train: Begin a new training; Test: Test at the last epoch)")
    parser.add_argument('-time', nargs='?', type=str, default=None,
                        help="Timestap of the checkpoint you want to load (Train: Load and continue to train; Test: Test at here)")
    parser.add_argument('-remark', nargs='?', type=str, default=None,
                        help="Input the remark you want to write at the name of experiment floder")
    parser.add_argument('-no_opt', action='store_true',
                        help='If you do not want to load the weight of optimizer or etc. choose it.')
    # parser.add_argument('-target_code', action='store_true',
    #                     help='use history code') 
    return parser

class Configuration(object):
    def __init__(self,parser, mode):
        super(Configuration,self).__init__()
        self.mode = mode
        args, extras = parser.parse_known_args()
        self.visual = args.visual if 'visual' in args else False
        self.save = args.save if 'save' in args else False
        self.with_tqdm = not args.no_tqdm if 'no_tqdm' in args else True

        with open('RootPath.json') as f:
            self.root = json5.load(f)
        if mode == 'train' and args.epoch is None:
            load_path, result_dir = None, None
        else:
            load_path, result_dir = self.__ensure_load_path(args)
        # if args.target_code:
        #     base = sys.path.pop(0)
        #     sys.path.append(os.path.join(result_dir,'code'))

        cfg_path = './config/{}.jsonc'.format(args.cfg_file)
        assert os.path.exists(cfg_path), 'Config file not exist!'
        with open(cfg_path) as f:
            dic = json5.load(f)
        if args.remark is not None:
            dic['system']['remark'] = args.remark.replace('\r','').replace('\n','')
        dic = self.__cfgChange(dic,extras)
        if mode == 'train':
            timestamp = datetime.now().strftime('%m%d_%H%M')
            if len(dic['system']['remark'])>0: timestamp += '_'+dic['system']['remark']
            result_dir = os.path.join(self.root[r'%RESULT%'], args.cfg_file, timestamp)
            self.__copy(args.cfg_file,dic,result_dir)

        self.system = dic['system']
        self.system['result_dir'] = result_dir
        self.system['load_path'] = load_path

        self.optimizer = dic['optimizer']
        self.optimizer['no_opt'] = args.no_opt

        self.dataset = dic['dataset']
        for key,val in dic['dataset'].items():
            if key in ('train','test'): continue
            self.dataset['train'][key] = val
            self.dataset['test'][key] = val

        for key in ('train', 'test'):
            if self.dataset[key]['direction'][0] == '%DATA%':
                self.dataset[key]['direction'][0] = self.root['%DATA%']
            self.dataset[key]['direction'] = os.path.join(*self.dataset[key]['direction'])

    def __cfgChange(self,dic,extras):
        extraParser = argparse.ArgumentParser()
        for arg in extras:
            if arg.startswith(("-", "--")):
                extraParser.add_argument(arg)
        chgDic = vars(extraParser.parse_args(extras))
        for arg_chain,val in chgDic.items():
            arg_list = arg_chain.split('.')
            change_flag = False
            for CLS in ('system','optimizer','dataset'):
                change_parent = dic[CLS]
                for arg in arg_list[:-1]:
                    if arg not in change_parent: break
                    change_parent = change_parent[arg]
                else:        
                    try: tmp = eval(val)
                    except:
                        tmp = val
                        if val.lower() in ['null','none']: tmp = None
                        elif val.lower() == 'true': tmp = True
                        elif val.lower() == 'false': tmp = False
                    if arg_list[-1] in change_parent:
                        change_parent[arg_list[-1]] = tmp
                        change_flag = True
            if not change_flag:
                print(yellow("Input q/Q to exist, any key to continue."))
                sys.stdout.flush()
                ins = input('#### {} do not exist, please check again ####'.format(yellow(CLS+'.'+arg_chain)))
                assert ins!='q' and ins!='Q'

        pprint(dic)
        return dic

    def __copy(self, cfg_name, cfg, result_dir):
        try:
            code_dir = os.path.join(result_dir,'code')
            os.makedirs(os.path.join(result_dir,'ckp'))
            os.makedirs(os.path.join(result_dir,'save'))
            os.makedirs(code_dir)

            tar_config_path = os.path.join(code_dir,'config')
            tar_model_path  = os.path.join(code_dir,'model')
            tar_data_path = os.path.join(code_dir,'dataset')
            os.makedirs(tar_config_path)
            os.makedirs(tar_model_path)
            os.makedirs(tar_data_path)


            with open(os.path.join(tar_config_path,cfg_name+'.jsonc'),'w') as f:
                json.dump(cfg,f)
            shutil.copy('RootPath.json',code_dir)
            shutil.copy('train.py',code_dir)
            shutil.copy('test.py',code_dir)
            shutil.copy(os.path.join('model',cfg['system']['net'][0]+'.py'),tar_model_path)
            shutil.copy(os.path.join('dataset',cfg['dataset']['file_name']+'.py'),tar_data_path)
            shutil.copytree('docker',os.path.join(code_dir,'docker'))
        except:
            key = input(yellow('\nDo you want to reserve this train(Default No)? y/n: '))
            if key == 'n' and os.path.exists(result_dir): 
                shutil.rmtree(result_dir)
            traceback.print_exc()
            sys.stdout.flush()


    def __ensure_load_path(self, args):
        direct = os.path.join(self.root[r'%RESULT%'], args.cfg_file)
        assert os.path.exists(direct), 'Net {} not exist'.format(args.cfg_file)

        target_timestap_list = os.listdir(direct)
        target_timestap = args.time
        if target_timestap is None:
            target_timestap_list.sort(key=lambda date: date[:9])
            target_timestap = target_timestap_list[-1]
        else:
            for item in target_timestap_list:
                if args.time == item[:len(args.time)]:
                    target_timestap = item
                    break
        direct = os.path.join(direct,target_timestap,'ckp')
        assert os.path.exists(direct), 'Timestamp {} not exist'.format(target_timestap)

        if args.epoch is None:
            epoches = os.listdir(direct)
            if 'best' in epoches:
                args.epoch = 'best'
            else:
                assert len(epoches) != 0, 'Epoch not exist'
                epoches = np.array(epoches, dtype=int)
                args.epoch = str(epoches.max())
        direct = os.path.join(direct, args.epoch)
        assert os.path.exists(direct), 'Epoch {} not exist'.format(args.epoch)

        return direct, os.path.join(self.root[r'%RESULT%'], args.cfg_file, target_timestap)


if __name__ == "__main__":
    args = base_args()
    cfg = Configuration(args,'train')