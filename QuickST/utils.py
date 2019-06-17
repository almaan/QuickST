#!/usr/bin/env python3

import pandas as pd
import numpy as np

def get_crd(ids):
    
    if isinstance(ids,pd.Index):
        ids = ids.tolist()
    
    ids = [[float(x) for x in y.split('x')] for y in ids]
    ids = np.array(ids).astype(np.float)
    
    return ids

def relative_frequencies(cnt, axis = 0):
    if len(cnt.shape) < 2:
        cnt = cnt.reshape(1,-1)
    if axis == 0:
        shape = (1,-1)
    else:
        shape = (-1,1)
    
    sm = np.sum(cnt, axis = axis)
    sm[sm == 0] = np.nan
    sm = sm.reshape(shape)
    rel = np.divide(cnt,sm)
    rel[np.isnan(rel)] = 0.0
    
    return rel