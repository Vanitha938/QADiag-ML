import copy,random
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
from .model import QADiagML
from .data import BiomarkerPreprocessor,assert_disjoint_subjects,STREAMS
from .metrics import classification_metrics

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def tensors(features,y=None):
    values=[torch.from_numpy(features[k]) for k in STREAMS]
    return TensorDataset(*(values+([torch.tensor(y,dtype=torch.long)] if y is not None else [])))

def train_model(model,train_feat,y_train,val_feat,y_val,epochs=40,batch=32,lr=.001,weight_decay=.0001,patience=9,device='cpu'):
    model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=4,factor=.5)
    counts=np.bincount(y_train,minlength=3);weights=torch.tensor(len(y_train)/(3*np.maximum(counts,1)),dtype=torch.float32,device=device)
    loader=DataLoader(tensors(train_feat,y_train),batch_size=batch,shuffle=True)
    val=[torch.from_numpy(val_feat[k]).to(device) for k in STREAMS];labels=torch.tensor(y_val,device=device)
    best=float('inf');state=None;wait=0;history=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for *xs,ys in loader:
            inp={k:x.to(device) for k,x in zip(STREAMS,xs)};ys=ys.to(device)
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(inp),ys,weight=weights);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():vl=torch.nn.functional.cross_entropy(model(dict(zip(STREAMS,val))),labels,weight=weights).item()
        scheduler.step(vl);history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':vl})
        if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());wait=0
        else:
            wait+=1
            if wait>=patience:break
    model.load_state_dict(state);return model,history

def predict(model,feat,device='cpu'):
    model.eval();model.to(device)
    with torch.no_grad():return model({k:torch.from_numpy(feat[k]).to(device) for k in STREAMS}).softmax(-1).cpu().numpy()

def cross_validate(df,cfg,folds=5,seeds=(42,43,44,45,46),epochs=40,device='cpu'):
    """Outer stratified patient-level folds; independent train-derived validation subsets."""
    from sklearn.model_selection import StratifiedShuffleSplit
    output=[];y=df['_label'].to_numpy();ids=df[cfg['subject_id']].astype(str)
    if ids.duplicated().any():raise ValueError('Expected one index visit per participant')
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    for fold,(train_idx,test_idx) in enumerate(skf.split(df,y),1):
        tr=df.iloc[train_idx];test=df.iloc[test_idx];assert_disjoint_subjects(tr,test,id_col=cfg['subject_id'])
        split=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=fold)
        a,b=next(split.split(tr,tr['_label']))
        fit,val=tr.iloc[a],tr.iloc[b];assert_disjoint_subjects(fit,val,test,id_col=cfg['subject_id'])
        for seed in seeds:
            seed_all(seed);pre=BiomarkerPreprocessor(cfg).fit(fit)
            model=QADiagML(pre.dimensions)
            model,h=train_model(model,pre.transform(fit),fit['_label'].to_numpy(),pre.transform(val),val['_label'].to_numpy(),epochs=epochs,device=device)
            prob=predict(model,pre.transform(test),device)
            output.append({'fold':fold,'seed':seed,'metrics':classification_metrics(test['_label'].to_numpy(),prob),'history':h})
    return output
