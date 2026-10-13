import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,brier_score_loss

def ece(y,prob,bins=15):
    conf=prob.max(1);correct=(prob.argmax(1)==y);edges=np.linspace(0,1,bins+1);total=0.
    for j in range(bins):
        mask=(conf>=edges[j]) & (conf<edges[j+1] if j<bins-1 else conf<=edges[j+1])
        if mask.sum():total+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return float(total)

def classification_metrics(y,prob):
    y=np.asarray(y);prob=np.asarray(prob);pred=prob.argmax(1)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],average='macro',zero_division=0)
    try:auc=roc_auc_score(y,prob,labels=[0,1,2],multi_class='ovr',average='macro')
    except ValueError:auc=float('nan')
    return dict(accuracy=float(accuracy_score(y,pred)),precision_macro=float(precision),recall_macro=float(recall),f1_macro=float(f1),auroc_ovr=float(auc),ece=ece(y,prob),brier=float(np.mean(np.sum((prob-np.eye(3)[y])**2,axis=1))),confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())

def bootstrap_ci(y,prob,metric='auroc_ovr',repeats=500,seed=42):
    rng=np.random.default_rng(seed);vals=[];n=len(y)
    for _ in range(repeats):
        idx=rng.choice(n,n,replace=True);v=classification_metrics(np.asarray(y)[idx],np.asarray(prob)[idx])[metric]
        if np.isfinite(v):vals.append(v)
    return np.quantile(vals,[.025,.975]).tolist() if vals else [float('nan')]*2
