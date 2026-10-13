#!/usr/bin/env python3
"""Executable survival-model smoke validation; real cohort landmarks require adjudication."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from qadiag_ml.model import QADiagML,LongitudinalQADiag
from qadiag_ml.longitudinal import train_survival,longitudinal_collate

def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--epochs',type=int,default=3);p.add_argument('--out',default='outputs/longitudinal_smoke.json');args=p.parse_args()
    if not args.synthetic:p.error('Real-dataset longitudinal preparation requires cohort-specific event adjudication. Use --synthetic to test the executable survival pipeline.')
    torch.set_num_threads(1);rng=np.random.default_rng(42);n=24;d={'clinical':6,'imaging':12,'bio_genetic':4}
    features={k:rng.normal(size=(n*3,v)).astype('float32') for k,v in d.items()}
    records=[{'indices':[i*3,i*3+1],'times':[0.,6.]} for i in range(n)]
    targets=pd.DataFrame({'event':np.array([i%2 for i in range(n)]),'event_interval':np.array([i%3 for i in range(n)]),'last_complete':np.array([i%3 for i in range(n)])})
    model=LongitudinalQADiag(QADiagML(d));losses=train_survival(model,records,features,targets,epochs=args.epochs)
    x,t,mask=longitudinal_collate(records,features)
    with torch.no_grad():risk=model(x,t,mask)['risk'].numpy().tolist()
    output={'warning':'Synthetic-only functional validation; no clinical evidence','loss_history':losses,'sample_risks':risk[:3]}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2));print('Saved',path)
if __name__=='__main__':main()
