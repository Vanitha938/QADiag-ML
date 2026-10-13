#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qadiag_ml.data import make_synthetic,load_cohort,make_classification_table,BiomarkerPreprocessor
from qadiag_ml.baselines import sklearn_baseline,concatenate
from qadiag_ml.metrics import classification_metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

def main():
    a=argparse.ArgumentParser();a.add_argument('--csv');a.add_argument('--schema');a.add_argument('--synthetic',action='store_true');a.add_argument('--out',default='outputs/baselines.json');args=a.parse_args()
    if args.synthetic:df,cfg=make_synthetic();print('SYNTHETIC TEST - NOT SCIENTIFIC EVIDENCE')
    elif args.csv and args.schema:df,cfg=load_cohort(args.csv,args.schema)
    else:a.error('Provide dataset or --synthetic')
    df=make_classification_table(df,cfg);y=df['_label'].values;results=[]
    for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y),1):
        pre=BiomarkerPreprocessor(cfg).fit(df.iloc[tr]);x=concatenate(pre.transform(df.iloc[tr]));xt=concatenate(pre.transform(df.iloc[te]))
        for name in ['logistic','random_forest','gradient_boosting','mlp']:
            model=sklearn_baseline(name);model.fit(x,y[tr]);prob=model.predict_proba(xt);results.append(dict(fold=f,model=name,metrics=classification_metrics(y[te],prob)))
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(results,indent=2));print('Saved',p)
if __name__=='__main__':main()
