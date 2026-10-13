"""MCI cohort temporal assembly without future-visit feature leakage."""
import numpy as np
import pandas as pd

def progression_targets(df,id_col,time_col,diagnosis_col,index_time_col=None):
    """Return index records with event time and observation time in months.
    Only observed progression to AD after MCI baseline qualifies as event.
    Other diagnostic routes require cohort-specific adjudication.
    """
    rows=[]
    for sid,group in df.groupby(id_col):
        g=group.sort_values(time_col);dx=g[diagnosis_col].astype(str).str.upper().str.strip()
        base_idx=np.where(dx.isin(['MCI','EMCI','LMCI']))[0]
        if not len(base_idx):continue
        start=int(base_idx[0]);g=g.iloc[start:];dx=dx.iloc[start:]
        start_time=float(g.iloc[0][time_col]);later=g.iloc[1:];conversions=later[later[diagnosis_col].astype(str).str.upper().eq('AD')]
        if len(conversions):event_time=float(conversions.iloc[0][time_col])-start_time;event=1
        else:event_time=float(g.iloc[-1][time_col])-start_time;event=0
        if event_time<0:continue
        intervals=[6,12,24];interval=next((k for k,h in enumerate(intervals) if event_time<=h),2)
        last_complete=sum(event_time>=h for h in intervals)-1
        rows.append(dict(subject_id=str(sid),event=event,event_time=event_time,event_interval=interval,last_complete=min(last_complete,2)))
    return pd.DataFrame(rows)

def longitudinal_collate(records,features,max_visits=None):
    """records: list of visit indices and elapsed months ending at prediction index."""
    from .data import STREAMS
    import torch
    n=len(records);length=max_visits or max(len(r['indices']) for r in records)
    t=torch.zeros(n,length);mask=torch.zeros(n,length,dtype=torch.bool)
    inp={k:torch.zeros(n,length,features[k].shape[1]) for k in STREAMS}
    for i,r in enumerate(records):
        idx=r['indices'][-length:];times=r['times'][-length:];l=len(idx)
        if not l:raise ValueError('Zero-visit participant')
        t[i,:l]=torch.tensor(times,dtype=torch.float32);mask[i,:l]=True
        for k in STREAMS:inp[k][i,:l]=torch.as_tensor(features[k][idx],dtype=torch.float32)
    return inp,t,mask

def train_survival(model,records,features,targets,epochs=25,lr=.001,device='cpu'):
    from .model import survival_nll
    import torch
    model.to(device);optim=torch.optim.Adam(model.parameters(),lr=lr)
    x,t,m=longitudinal_collate(records,features);x={k:v.to(device) for k,v in x.items()};t=t.to(device);m=m.to(device)
    event=torch.tensor(targets['event'].to_numpy(),device=device,dtype=torch.bool)
    interval=torch.tensor(targets['event_interval'].to_numpy(),device=device,dtype=torch.long)
    complete=torch.tensor(targets['last_complete'].to_numpy(),device=device,dtype=torch.long)
    history=[]
    for _ in range(epochs):
        model.train();optim.zero_grad();haz=model(x,t,m)['hazard'];loss=survival_nll(haz,interval,event,complete)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();history.append(float(loss.item()))
    return history
