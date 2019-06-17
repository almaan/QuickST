#!/usr/bin/env python3


from typing import Dict, List
import pandas as pd
import numpy as np

def join_samples(samples : List[pd.Index]) -> Dict:
    
    """Join multiple samples together"""
    
    if not isinstance(samples,list):
        samples = [samples]
    
    allcols = pd.Index([])
    
    idx_list = list()
    new_idx = list()
    pos_idx = np.array([0])
    rename = lambda x: ''.join(['sample',str(num),'_',str(x)])
    for num,s in enumerate(samples):
        allcols = allcols.union(s.columns)
        idx_list.append(s.index)
        new_idx += [rename(x) for x in s.index]
        pos_idx = np.append(pos_idx,s.index.shape[0])
    
    new_idx = pd.Index(new_idx)
    n_cols = allcols.shape[0]
    cumsum = np.cumsum(pos_idx)
    jmat = np.zeros((cumsum[-1],n_cols))
    jmat = pd.DataFrame(jmat,
                        columns = allcols,
                        index = new_idx)
    
    
    for pos,s in enumerate(samples):
        jmat.loc[cumsum[pos]:cumsum[pos+1],s.columns] = s.values
        samples[pos] = None
    
    joint = dict(joint_matrix = jmat,
                 idx_list = idx_list)    
    return joint
    
def split_samples(joint_matrix : pd.DataFrame,
                  idx_list : List[pd.Index],
                  ) -> List:
    
    """Split joined samples
    
    Decomposes a joint matrix into the respective
    constituents.
    
    """
    
    
    
    samples = list()
    s_start = 0
    for num in range(len(idx_list)):
        s_end = s_start + idx_list[num].shape[0]
        tmp = joint_matrix.iloc[s_start:s_end,:]
        tmp.index = idx_list[num]
        samples.append(tmp)
        s_start = s_end
        
    return samples
    