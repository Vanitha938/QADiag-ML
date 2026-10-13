#!/usr/bin/env python3
import argparse,time,sys,json,tracemalloc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from qadiag_ml.model import QADiagML
from qadiag_ml.baselines import NeuralBaseline

def main():
    a=argparse.ArgumentParser();a.add_argument('--dimensions',nargs='+',type=int,default=[16,32,64,128]);a.add_argument('--repeats',type=int,default=3);a.add_argument('--out',default='outputs/scaling.json');args=a.parse_args();results=[]
    torch.set_num_threads(1)
    for d in args.dimensions:
        for name in ['qadiag','fusion']:
            dims={'clinical':d,'imaging':d,'bio_genetic':d};model=QADiagML(dims) if name=='qadiag' else NeuralBaseline(dims)
            x={k:torch.randn(32,d) for k in dims};opt=torch.optim.Adam(model.parameters(),lr=.001)
            t0=time.perf_counter();tracemalloc.start()
            for _ in range(args.repeats):
                opt.zero_grad();loss=model(x).square().mean();loss.backward();opt.step()
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            results.append({'dimension_per_modality':d,'model':name,'seconds_per_step':(time.perf_counter()-t0)/args.repeats,'python_peak_memory_bytes':peak,'parameters':sum(v.numel() for v in model.parameters())})
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(results,indent=2));print(p)
    print('Memory is Python allocation ONLY, not total process/GPU memory; use platform profiler for research memory results.')
if __name__=='__main__':main()
