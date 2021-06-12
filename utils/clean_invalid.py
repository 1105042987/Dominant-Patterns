import sys,os
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import json5
import shutil
from glob import glob
from docker.tool import yellow

with open('RootPath.json') as f:
    root = json5.load(f)[r'%RESULT%']
assert len(sys.argv)>1, 'Project Name Needed!'
tars = glob(os.path.join(root,sys.argv[1],'*'))
print('')
for path in tars:
    if len(glob(os.path.join(path,'*.txt')))==0:
        if len(sys.argv)==3 and sys.argv[2]=='D':
            shutil.rmtree(path)
        else:
            print(yellow(path))
if len(sys.argv)==2:
    print('### Above is invaild floder ###')
    print('Add second paramter \'D\' behind project name to delete them.')
    print('Such as:', yellow(f'$ python clean_invaild.py {sys.argv[1]} D'))


    
