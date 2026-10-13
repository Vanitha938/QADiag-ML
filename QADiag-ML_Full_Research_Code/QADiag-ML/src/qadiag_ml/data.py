"""Cohort ingestion, transparent schema mapping and train-only preprocessing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold

STREAMS=('clinical','imaging','bio_genetic')
LABELS={'CN':0,'NL':0,'NORMAL':0,'MCI':1,'EMCI':1,'LMCI':1,'AD':2,'EARLY AD':2}

def load_cohort(path,schema):
    """Schema JSON: subject_id, diagnosis, optional visit_time, site, and stream -> lists of column names."""
    cfg=json.loads(Path(schema).read_text()) if isinstance(schema,(str,Path)) else schema
    df=pd.read_csv(path,low_memory=False)
    required=[cfg['subject_id'],cfg['diagnosis']]+sum((cfg[s] for s in STREAMS),[])
    missing=[x for x in required if x not in df]
    if missing:raise ValueError(f'Missing schema columns: {missing}')
    label=df[cfg['diagnosis']].astype(str).str.upper().str.strip().map({**LABELS,**{k.upper():v for k,v in cfg.get('label_map',{}).items()}})
    df=df.loc[label.notna()].copy();df['_label']=label.loc[df.index].astype(int)
    if df[cfg['subject_id']].isna().any():raise ValueError('Subject ID missing')
    return df,cfg

def make_classification_table(df,cfg):
    """Use earliest visit per participant, never select using outcomes after baseline."""
    if cfg.get('visit_time') in df:df=df.sort_values([cfg['subject_id'],cfg['visit_time']])
    return df.drop_duplicates(cfg['subject_id'],keep='first').reset_index(drop=True)

def assert_disjoint_subjects(*frames,id_col):
    sets=[set(f[id_col].astype(str)) for f in frames]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            if sets[i]&sets[j]:raise ValueError(f'Subject leakage between partitions {i} and {j}')

def check_adni_tadpole_overlap(adni_ids,tadpole_ids,heldout_ids):
    forbidden=set(map(str,heldout_ids))&set(map(str,tadpole_ids))
    return {'shared_count':len(set(map(str,adni_ids))&set(map(str,tadpole_ids))), 'excluded_tadpole_ids':sorted(forbidden)}

class BiomarkerPreprocessor:
    def __init__(self,schema,demographic_adjust=False,site_min=8):
        self.cfg=schema;self.demographic_adjust=demographic_adjust;self.site_min=site_min
        self.models={};self.scalers={};self.site_params={};self.fitted=False
    def fit(self,frame):
        self.models={};self.scalers={};self.site_params={};self.columns={}
        site_col=self.cfg.get('site');sites=frame[site_col].fillna('_unknown').astype(str).to_numpy() if site_col in frame else None
        for stream in STREAMS:
            cols=self.cfg[stream];self.columns[stream]=cols
            raw=frame[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
            if np.isnan(raw).all(axis=0).any():raise ValueError(f'All-missing columns in {stream}: {[c for c,m in zip(cols,np.isnan(raw).all(axis=0)) if m]}')
            imp=IterativeImputer(max_iter=10,random_state=42,skip_complete=False) if raw.shape[1]>1 else SimpleImputer(strategy='median')
            filled=imp.fit_transform(raw);scaler=StandardScaler().fit(filled);scaled=scaler.transform(filled)
            self.models[stream]=imp;self.scalers[stream]=scaler
            if stream=='imaging' and sites is not None:
                self.site_params[stream]={}
                for site in np.unique(sites):
                    subset=scaled[sites==site]
                    if len(subset)>=self.site_min:self.site_params[stream][site]=(subset.mean(0),np.maximum(subset.std(0),1e-6))
        self.fitted=True;return self
    def transform(self,frame):
        if not self.fitted:raise RuntimeError('Preprocessor not fitted')
        result={};site_col=self.cfg.get('site');sites=frame[site_col].fillna('_unknown').astype(str).to_numpy() if site_col in frame else None
        for stream in STREAMS:
            raw=frame[self.columns[stream]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
            mask=np.isnan(raw).astype(np.float32)
            values=self.scalers[stream].transform(self.models[stream].transform(raw))
            if stream=='imaging' and sites is not None:
                for site,(mu,sd) in self.site_params.get(stream,{}).items():
                    ix=sites==site;values[ix]=(values[ix]-mu)/sd
            result[stream]=np.concatenate((values,mask),axis=-1).astype('float32')
        return result
    def fit_transform(self,frame):return self.fit(frame).transform(frame)
    @property
    def dimensions(self):return {s:2*len(self.columns[s]) for s in STREAMS}

def make_synthetic(n=180,seed=42):
    """Synthetic fixtures ONLY, never use for scientific performance claims."""
    rng=np.random.default_rng(seed);y=np.repeat(np.arange(3),n//3);n=len(y)
    df=pd.DataFrame({'RID':[f'S{j:04d}' for j in range(n)],'DX':np.array(['CN','MCI','AD'])[y],
        'site':rng.choice(['A','B','C'],n),'month':np.zeros(n),'_label':y})
    cfg={'subject_id':'RID','diagnosis':'DX','visit_time':'month','site':'site'}
    for stream,d in [('clinical',6),('imaging',12),('bio_genetic',5)]:
        cfg[stream]=[f'{stream}_{i}' for i in range(d)]
        for i,col in enumerate(cfg[stream]):
            v=rng.normal(loc=y*.4*(i%3==0),size=n);v[rng.random(n)<.12]=np.nan;df[col]=v
    return df,cfg
