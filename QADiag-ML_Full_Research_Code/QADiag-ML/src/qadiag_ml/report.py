import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import classification_metrics

def save_cv_report(results,destination):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    (p/'runs.json').write_text(json.dumps(results,indent=2))
    data=pd.DataFrame([{'fold':v['fold'],'seed':v['seed'],**{k:x for k,x in v['metrics'].items() if k!='confusion_matrix'}} for v in results])
    data.to_csv(p/'run_metrics.csv',index=False)
    summary=data.groupby('fold').mean(numeric_only=True).drop(columns=['seed'],errors='ignore')
    summary.to_csv(p/'fold_metrics.csv')
    summary.agg(['mean','std']).to_csv(p/'summary.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for v in results[:5]:ax.plot([h['epoch'] for h in v['history']],[h['val_loss'] for h in v['history']],alpha=.7)
    ax.set(xlabel='Epoch',ylabel='Validation loss',title='Diagnostic Validation Learning Curves');fig.tight_layout();fig.savefig(p/'learning_curves.png',dpi=300);plt.close(fig)
    return summary
