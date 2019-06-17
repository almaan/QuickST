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

def control_lists(list1 : list,
                  list2 : list) -> bool:
    """
    Control if the sorting of two lists
    are optimal in terms of similarity.
    Returns true if lists seems to be 
    optimally sorted and false otherwise.
    
    
    Arguments
    ---------
        list1 : list
            first list to be compared
        list2 : list
            second list to be compared
    
    Returns
    ------
    
    """
    
    def maxham(s1,s2):
        abscore = 0.0
        if len(s1) > len(s2):
            major,minor = s1,s2
        else:
            major, minor = s2,s1
        
        w = len(minor)
        for pos in range(len(major) - w +1):
            tscore = hamming(major[pos:pos + w],minor)
            if tscore > abscore:
                abscore = tscore
                
        return abscore
    
    def hamming(x,y):
        hd = 0.0
        for (xs,ys) in zip(x,y):
            if xs == ys:
                hd += 1
                
        return hd

    if not len(list1) == len(list2):
        return False
    else:
        smat = np.zeros((len(list1),len(list2)))
        for l1 in range(len(list1)):
            for l2 in range(len(list2)):
                smat[l1,l2] = maxham(list1[l1],list2[l2])
        
        diag = np.diag(smat)
        mx = np.max(smat, axis = 1)
        
        return (mx == diag).all()
    
