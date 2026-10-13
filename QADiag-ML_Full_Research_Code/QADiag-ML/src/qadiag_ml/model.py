"""Exact dimensional implementation of QADiag-ML, on classical hardware."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, d, hidden=64, out=32, dropout=.2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,out),nn.LayerNorm(out))
    def forward(self,x):return self.net(x)

class InteractionCandidates(nn.Module):
    """Projected products for CI, CB, IB and CIB."""
    def __init__(self,d=32):
        super().__init__()
        self.proj=nn.ModuleList([nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(k)]) for k in [2,2,2,3]])
    def forward(self,c,i,b):
        combos=((c,i),(c,b),(i,b),(c,i,b))
        return torch.stack([torch.stack([layer(x) for layer,x in zip(layers,xs)],dim=0).prod(0) for layers,xs in zip(self.proj,combos)],dim=1)

class AngleMap(nn.Module):
    def __init__(self,d=32,scale=1.):
        super().__init__();self.proj=nn.Linear(d,4);self.scale=nn.Parameter(torch.tensor(float(scale)))
    def forward(self,x):
        theta=self.proj(x)*self.scale
        return torch.stack((torch.cos(theta),torch.sin(theta)),dim=-1).reshape(x.shape[0],16,2)

class MPS(nn.Module):
    """16 sites, physical 2, bond 8; output 32 in final core. Core params=2320."""
    def __init__(self,sites=16,physical=2,bond=8,out=32):
        super().__init__();self.sites=sites;self.physical=physical;self.bond=bond;self.out=out
        shape=[(physical,bond)]+[(bond,physical,bond)]*(sites-2)+[(bond,physical,out)]
        self.cores=nn.ParameterList([nn.Parameter(torch.randn(*s)*(.15/math.sqrt(bond))) for s in shape])
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.sites,self.physical):raise ValueError('Expected [N,16,2]')
        h=torch.einsum('np,pr->nr',x[:,0],self.cores[0])
        for k in range(1,self.sites-1):
            h=torch.einsum('na,np,apr->nr',h,x[:,k],self.cores[k])
            # Normalize intermediates to improve numerical stability without changing direction
            h=F.layer_norm(h,(h.shape[-1],))
        return torch.einsum('na,np,apd->nd',h,x[:,-1],self.cores[-1])
    @property
    def core_parameters(self):return sum(p.numel() for p in self.cores)

class PrototypeKernel(nn.Module):
    def __init__(self,d=32,k=32,bandwidth=1.,temperature=1.):
        super().__init__();self.keys=nn.Parameter(torch.randn(k,d)*.1);self.values=nn.Parameter(torch.randn(k,d)*.1)
        self.log_bandwidth=nn.Parameter(torch.tensor(float(bandwidth)).log())
        self.log_temperature=nn.Parameter(torch.tensor(float(temperature)).log())
        self.out=nn.Linear(d,d)
    def forward(self,x):
        xx=F.normalize(x,dim=-1,eps=1e-8);keys=F.normalize(self.keys,dim=-1,eps=1e-8)
        dist=torch.cdist(xx,keys).square();bw=self.log_bandwidth.exp().clamp(min=.05,max=20)
        weights=F.softmax(-dist/(2*bw*bw)/self.log_temperature.exp().clamp(min=.05,max=20),dim=-1)
        return x+self.out(weights@self.values)

class QADiagML(nn.Module):
    def __init__(self,input_dims,dropout=.2,bond=8,use_angle=True,use_mps=True,use_kernel=True,use_gate=True,candidates=(0,1,2,3)):
        super().__init__();self.input_dims=dict(input_dims);self.candidates=tuple(candidates)
        self.enc=nn.ModuleDict({k:ModalityEncoder(v,dropout=dropout) for k,v in self.input_dims.items()})
        if set(self.input_dims)!={'clinical','imaging','bio_genetic'}:raise ValueError('Input dims must have clinical, imaging, bio_genetic')
        self.interactions=InteractionCandidates();self.angle=AngleMap();self.use_angle=use_angle
        self.plain_map=nn.Linear(32,8);self.use_mps=use_mps
        self.mps=MPS(bond=bond);self.flat_interaction=nn.Sequential(nn.Flatten(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
        self.kernel=PrototypeKernel();self.use_kernel=use_kernel;self.use_gate=use_gate
        self.direct=nn.Linear(96,32);self.gate=nn.Linear(64,32)
        self.attention=nn.Linear(32,32);self.classifier=nn.Linear(32,3)
        self.dropout=nn.Dropout(dropout)
    def forward(self,inputs,return_details=False):
        c,i,b=(self.enc[key](inputs[key]) for key in ('clinical','imaging','bio_genetic'))
        candidates=self.interactions(c,i,b);mask=torch.zeros_like(candidates)
        for idx in self.candidates:mask[:,idx]=candidates[:,idx]
        candidates=mask
        local=self.angle(candidates) if self.use_angle else F.normalize(self.plain_map(candidates).reshape(-1,4,4,2),dim=-1).reshape(-1,16,2)
        inter=self.mps(local) if self.use_mps else self.flat_interaction(local)
        refined=self.kernel(inter) if self.use_kernel else inter
        direct=self.direct(torch.cat((c,i,b),-1));gate=torch.sigmoid(self.gate(torch.cat((direct,refined),-1)))
        fused=gate*direct+(1-gate)*refined if self.use_gate else .5*(direct+refined)
        attention=F.softmax(self.attention(fused),dim=-1);representation=self.dropout(attention*fused)
        logits=self.classifier(representation)
        if return_details:return {'logits':logits,'probabilities':logits.softmax(-1),'candidates':candidates,'interaction':refined,'fused':fused,'representation':representation,'attention':attention,'gate':gate}
        return logits

class LongitudinalQADiag(nn.Module):
    def __init__(self,base,hidden=32):
        super().__init__();self.base=base;self.time_embed=nn.Linear(2,32)
        self.score=nn.Sequential(nn.Linear(32,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.hazards=nn.Linear(32,3)
    def forward(self,inputs,times,valid):
        n,t=times.shape;flat={key:val.reshape(n*t,-1) for key,val in inputs.items()}
        z=self.base(flat,return_details=True)['fused'].reshape(n,t,32)
        dt=torch.cat((torch.zeros_like(times[:,:1]),(times[:,1:]-times[:,:-1]).clamp_min(0)),1)
        time_features=torch.stack((dt/24.,torch.log1p(dt)),dim=-1)
        score=self.score(z+self.time_embed(time_features)).squeeze(-1).masked_fill(~valid,-1e9)
        if not valid.any(dim=1).all():raise ValueError('Each participant needs at least one valid visit')
        weights=F.softmax(score,dim=1);subject=(weights.unsqueeze(-1)*z).sum(1)
        hazard=torch.sigmoid(self.hazards(subject));risk=1-torch.cumprod(1-hazard,dim=-1)
        return {'hazard':hazard,'risk':risk,'attention':weights,'subject_embedding':subject}

def survival_nll(hazard,event_interval,event_observed,last_complete):
    """event_interval: 0/1/2 for converters; last_complete: -1/0/1/2 for censored; -1=no completed interval."""
    eps=1e-7;hazard=hazard.clamp(eps,1-eps);idx=torch.arange(3,device=hazard.device)[None,:]
    before=idx<event_interval[:,None];event=(idx==event_interval[:,None])
    observed=event_observed.bool();surv_mask=torch.where(observed[:,None],before,idx<=last_complete[:,None])
    ll=(surv_mask*(-torch.log1p(-hazard))).sum(1)
    ll=ll+torch.where(observed,-torch.log(hazard.gather(1,event_interval.clamp(0,2)[:,None]).squeeze(1)),torch.zeros_like(ll))
    return ll.mean()
