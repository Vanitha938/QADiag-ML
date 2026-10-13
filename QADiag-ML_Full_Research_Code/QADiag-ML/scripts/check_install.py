#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from qadiag_ml.model import MPS,QADiagML
m=MPS()
assert m.core_parameters==2320
print('QADiag-ML installed; Torch',torch.__version__,'MPS cores:',m.core_parameters)
