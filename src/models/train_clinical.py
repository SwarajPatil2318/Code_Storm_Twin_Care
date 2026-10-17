import argparse
from .clinical import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/processed');p.add_argument('--output',default='data/results/phase3/clinical');a=p.parse_args()
    print(train(a.root,a.output))
