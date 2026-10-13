import numpy as np
import torch
from qadiag_ml.model import QADiagML,MPS,LongitudinalQADiag,survival_nll
from qadiag_ml.data import make_synthetic,BiomarkerPreprocessor,assert_disjoint_subjects
from qadiag_ml.metrics import classification_metrics

def test_mps_parameter_count():
    m=MPS();assert m.core_parameters==2320
    x=torch.randn(5,16,2);assert m(x).shape==(5,32)

def test_end_to_end_gradients():
    torch.manual_seed(42);m=QADiagML({'clinical':12,'imaging':24,'bio_genetic':10})
    x={'clinical':torch.randn(6,12),'imaging':torch.randn(6,24),'bio_genetic':torch.randn(6,10)}
    out=m(x,True);assert out['candidates'].shape==(6,4,32)
    assert out['logits'].shape==(6,3);assert out['attention'].shape==(6,32)
    out['logits'].square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None) and m.mps.cores[0].grad is not None

def test_longitudinal_risk():
    base=QADiagML({'clinical':4,'imaging':6,'bio_genetic':2});model=LongitudinalQADiag(base)
    x={k:torch.randn(3,4,v) for k,v in base.input_dims.items()}
    t=torch.tensor([[0.,6.,12.,24.]]*3);valid=torch.tensor([[True,True,False,False]]*3)
    result=model(x,t,valid);risk=result['risk'];assert risk.shape==(3,3)
    assert (risk[:,1:]>=risk[:,:-1]).all()
    hazard=result['hazard'];loss=survival_nll(hazard,torch.tensor([0,1,2]),torch.tensor([True,False,True]),torch.tensor([0,1,2]));assert torch.isfinite(loss)

def test_preprocessing_and_splits():
    frame,cfg=make_synthetic(90);pre=BiomarkerPreprocessor(cfg).fit(frame.iloc[:60]);x=pre.transform(frame.iloc[60:]);assert len(x)==3
    assert all(np.isfinite(z).all() for z in x.values())
    assert x['imaging'].shape==(30,24)
    try:assert_disjoint_subjects(frame.iloc[:30],frame.iloc[:35],id_col='RID')
    except ValueError:pass
    else:raise AssertionError('Expected leakage detection')

def test_metrics():
    y=np.array([0,1,2]);p=np.eye(3)*.95+.05/3
    assert classification_metrics(y,p)['accuracy']==1
