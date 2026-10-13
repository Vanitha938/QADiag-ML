#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qadiag_ml.data import make_synthetic,BiomarkerPreprocessor
from qadiag_ml.model import QADiagML
from qadiag_ml.train import train_model,predict,seed_all
from qadiag_ml.metrics import classification_metrics
from sklearn.model_selection import train_test_split

def main():
    a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=10);a.add_argument('--out',default='outputs/ablations.json');args=a.parse_args()
    df,cfg=make_synthetic(180);a,b=train_test_split(df,test_size=.25,stratify=df['_label'],random_state=42);a,v=train_test_split(a,test_size=.2,stratify=a['_label'],random_state=42)
    results=[]
    for name,config in [('full',{}),('no_angle',{'use_angle':False}),('no_mps',{'use_mps':False}),('no_kernel',{'use_kernel':False}),('no_gate',{'use_gate':False}),('pair_only',{'candidates':(0,1,2)})]:
        seed_all(42);pre=BiomarkerPreprocessor(cfg).fit(a);model=QADiagML(pre.dimensions,**config)
        model,_=train_model(model,pre.transform(a),a['_label'].values,pre.transform(v),v['_label'].values,epochs=args.epochs)
        results.append({'model':name,'metrics':classification_metrics(b['_label'].values,predict(model,pre.transform(b)))})
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'warning':'Synthetic pipeline demonstration, not scientific evidence','runs':results},indent=2));print('Saved',p)
if __name__=='__main__':main()
