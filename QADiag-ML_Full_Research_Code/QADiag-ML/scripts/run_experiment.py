#!/usr/bin/env python3
"""Diagnostic cross-validation on local CSV/schema; synthetic smoke mode explicitly marked."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qadiag_ml.data import make_synthetic,load_cohort,make_classification_table
from qadiag_ml.train import cross_validate
from qadiag_ml.report import save_cv_report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv');parser.add_argument('--schema');parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]);parser.add_argument('--output',default='outputs/diagnostic');parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.synthetic:df,cfg=make_synthetic();print('SYNTHETIC SMOKE DATA: NOT SCIENTIFIC EVIDENCE')
    elif args.csv and args.schema:df,cfg=load_cohort(args.csv,args.schema)
    else:parser.error('Provide --csv and --schema, or --synthetic')
    df=make_classification_table(df,cfg)
    results=cross_validate(df,cfg,folds=args.folds,seeds=args.seeds,epochs=args.epochs,device=args.device)
    print(save_cv_report(results,args.output).to_string());print('Saved outputs:',args.output)
if __name__=='__main__':main()
