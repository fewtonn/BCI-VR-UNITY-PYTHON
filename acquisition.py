import numpy as np
from pylsl import StreamInlet, resolve_stream
from config import *

def normalize_sample(input_sample):
    arr=np.array(input_sample); return (arr-arr.min())/(arr.max()-arr.min()+1e-8)

class Sistema:
    def __init__(self): self.dt=4.506
