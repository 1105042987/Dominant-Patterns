import numpy as np

def yellow(string):
    if not isinstance(string,str):
        string = str(string)
    return f"\033[33m{string}\033[0m"

class meter(object):
    def __init__(self):
        super(meter,self).__init__()
        self.reset()

    def reset(self):
        self.cnt = {}
        self.dic = {}
    
    def add(self,out):
        if self.cnt == {}: 
            for key,val in out.items():
                try:
                    self.cnt[key] = len(val)
                except:
                    self.cnt[key] = 1
                self.dic[key] = np.sum(val)
        else: 
            for key,val in out.items():
                try:
                    self.cnt[key] += len(val)
                except:
                    self.cnt[key] += 1
                self.dic[key] += np.sum(val)

    def mean(self):
        dic = {}
        if self.cnt == 0:
            return {'None':None}
        for key, val in self.dic.items():
            dic[key] = self.dic[key]/self.cnt[key]
        return dic
