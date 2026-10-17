import argparse
from .forecasting import evaluate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/processed');p.add_argument('--output',default='data/results/phase3');p.add_argument('--dataset',choices=['all','chiron','perheart'],default='all');a=p.parse_args()
    print(evaluate(a.root,a.output,('chiron','perheart') if a.dataset=='all' else (a.dataset,)))
