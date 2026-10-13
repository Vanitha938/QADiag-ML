"""Attribution to the four actual candidate tensors, not to attention weights."""
import itertools
import numpy as np
import torch
from .data import STREAMS

def gradient_input(model,features,target=None,device='cpu'):
    model.eval().to(device)
    x={k:torch.tensor(features[k],device=device) for k in STREAMS}
    saved={}
    def hook(module,inputs,output):
        output.retain_grad();saved['candidates']=output
    h=model.interactions.register_forward_hook(hook)
    try:
        details=model(x,return_details=True);logits=details['logits']
        cls=logits.argmax(-1) if target is None else torch.full((len(logits),),target,device=device,dtype=torch.long)
        score=logits.softmax(-1).gather(1,cls[:,None]).sum()
        model.zero_grad(set_to_none=True);score.backward()
        c=saved['candidates'];result=(c.detach()*c.grad.detach()).sum(-1)
        return result.cpu().numpy()
    finally:h.remove()

def stability(fold_relevance,topk=2):
    scores=np.asarray(fold_relevance,dtype=float)
    if scores.ndim!=2 or scores.shape[1]!=4:raise ValueError('Expected folds x 4 scores')
    v=scores.var(0);s=1-v/(v.max()+1e-8)
    from scipy.stats import spearmanr
    jaccard=[];rank=[]
    for i,j in itertools.combinations(range(len(scores)),2):
        a=set(np.argsort(scores[i])[-topk:]);b=set(np.argsort(scores[j])[-topk:]);jaccard.append(len(a&b)/len(a|b))
        r=spearmanr(scores[i],scores[j]).statistic;rank.append(r)
    return {'normalized_stability':s.tolist(),'jaccard':float(np.mean(jaccard)) if jaccard else None,'spearman':float(np.nanmean(rank)) if rank else None}

def integrated_gradients_candidates(model,features,steps=32):
    """Captum candidate-level IG via a forward hook that replaces the interaction tensors."""
    try:from captum.attr import IntegratedGradients
    except ImportError as exc:raise ImportError('pip install captum') from exc
    device=next(model.parameters()).device;model.eval()
    xs={k:torch.tensor(features[k],device=device) for k in STREAMS}
    with torch.no_grad():
        c=model.interactions(*[model.enc[k](xs[k]) for k in STREAMS]).detach()
        cls=model(xs).argmax(-1)
    def prediction(candidates):
        def replace(module,inputs,output):return candidates
        h=model.interactions.register_forward_hook(replace)
        try:return model(xs).softmax(-1)
        finally:h.remove()
    attribution=IntegratedGradients(prediction).attribute(c,baselines=torch.zeros_like(c),target=cls,n_steps=steps)
    return attribution.detach().sum(-1).cpu().numpy()

def shap_candidates(model,features,background=None,nsamples=100):
    """SHAP KernelExplainer on flattened candidate vectors (computationally costly)."""
    try:import shap
    except ImportError as exc:raise ImportError('pip install shap') from exc
    device=next(model.parameters()).device;model.eval()
    xs={k:torch.tensor(features[k],device=device) for k in STREAMS}
    with torch.no_grad():
        original=model.interactions(*[model.enc[k](xs[k]) for k in STREAMS]).detach()
        target=model(xs).argmax(-1)
    def predict(flat):
        out=[]
        for idx,row in enumerate(flat):
            candidate=torch.tensor(row,device=device,dtype=original.dtype).reshape(1,4,32)
            subset={k:v[idx:idx+1] for k,v in xs.items()}
            h=model.interactions.register_forward_hook(lambda m,i,o:candidate)
            try:
                with torch.no_grad():out.append(model(subset).softmax(-1).cpu().numpy()[0])
            finally:h.remove()
        return np.array(out)
    arr=original.cpu().numpy().reshape(len(original),-1);bg=np.zeros((1,128)) if background is None else background
    explainer=shap.KernelExplainer(predict,bg)
    return explainer.shap_values(arr,nsamples=nsamples)
