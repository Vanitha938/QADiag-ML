#!/usr/bin/env python3
"""Train four nonlinear baseline architectures with matched split on synthetic fixture."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sklearn.model_selection import train_test_split
from qadiag_ml.data import make_synthetic,BiomarkerPreprocessor
from qadiag_ml.baselines import NeuralBaseline
from qadiag_ml.train import train_model,predict,seed_all
from qadiag_ml.metrics import classification_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--epochs',type=int,default=3);p.add_argument('--out',default='outputs/neural_baselines.json');args=p.parse_args()
    if not args.synthetic:p.error('This runner is a synthetic integration demonstration; adapt matching data preparation for real cohorts.')
    df,cfg=make_synthetic();fit,test=train_test_split(df,test_size=.2,stratify=df['_label'],random_state=42)
    fit,val=train_test_split(fit,test_size=.2,stratify=fit['_label'],random_state=42)
    pre=BiomarkerPreprocessor(cfg).fit(fit);out=[]
    for kind in ['fusion','transformer','gnn','late_fusion']:
        seed_all(42);model=NeuralBaseline(pre.dimensions,kind=kind)
        model,h=train_model(model,pre.transform(fit),fit['_label'].values,pre.transform(val),val['_label'].values,epochs=args.epochs)
        out.append({'model':kind,'metrics':classification_metrics(test['_label'].values,predict(model,pre.transform(test)))})
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'warning':'Synthetic only','runs':out},indent=2));print('Saved',path)
if __name__=='__main__':main()
