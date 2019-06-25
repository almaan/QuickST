#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class STsection:
    def __init__(self,
                 cnt_pth,
                 meta_pth = None,
                 ):
    
        self._cnt = pd.read_csv(cnt_pth,
                                sep = '\t',
                                header = 0,
                                index_col = 0)

        if meta_pth:
            self._meta = pd.read_csv(meta_pth,
                                     sep = '\t',
                                     header = 0,
                                     index_col = 0)
            
            self.xcol = self._meta.columns[[ 'x' == x[0] for x in self._meta.columns]][0]
            self.ycol = self._meta.columns[[ 'y' == x[0] for x in self._meta.columns]][0]
            
            xc = np.round(self._meta[self.xcol].values).astype(int)
            yc = np.round(self._meta[self.ycol].values).astype(int)
            
            in_meta = pd.Index(['x'.join([str(x),str(y)]) for x,y in zip(xc,yc)])
            inter = self._cnt.index.intersection(in_meta)
            
            self._cnt = self._cnt.reindex(inter)
            self._meta = self._meta.reindex(inter)
            
            self.has_meta = True
        else:
            self.has_meta = False
            self._meta = None
            self._x = np.array([float(x.split('x')[0]) for x \
                                in self.cnt.index.tolist()])
            self._y = np.array([float(x.split('x')[1]) for x \
                                in self.cnt.index.tolist()])
            
        self._update_identifiers()
        
    def _update_identifiers(self,):
        
        if self.has_meta and 'patient' in self._meta.columns:
            self.patient = self._meta['patient'].astype(str).iloc[0]
        else:
            self.patient = '_patient_unknown_'
        
        if self.has_meta and 'replicate' in self._meta.columns:
            self.replicate = self._meta['replicate'].astype(str).iloc[0]
        else:
            self.replicate = '_replicate_unknown_'
        
        if self.has_meta:
            self.K = self.meta.shape[1]
            
        self.genes = self._cnt.columns
        self.spots = self._cnt.index

        self.G = self.cnt.shape[1]
        self.S = self.cnt.shape[0]
        
        
    @property
    def cnt(self,):
        return self._cnt
    
    @cnt.setter
    def cnt(self,val):
        raise RuntimeError
        
    @property
    def meta(self,):
        return self._meta
    
    @meta.setter
    def meta(self,val):
        raise RuntimeError
    
    @property
    def x(self,):
        if self.has_meta:
            return self._meta[self.xcol].values
        else:
            return self._x
            
    @property
    def y(self,):
        if self.has_meta:
            return self._meta[self.ycol].values
        else:
            return self._y
    
    @property
    def ncnt(self,):
        return self._cnt / self._cnt.sum(axis=1).reshape(-1,1)
        
    def update_meta(self,data, colnames = None):
        if self.has_meta:
            if isinstance(data, np.ndarray):
                if not isinstance(colnames,list):
                    self._meta[colnames] = data
                else:
                    for k,name in enumerate(colnames):
                        self._meta[name] = data[:,k]
                        
                self._update_identifiers()
            
            elif isinstance(data, pd.DataFrame):
                colnames = data.columns.values
                for k,name in enumerate(colnames):                
                    self._meta[name] = data.iloc[:,k]
                self._update_identifiers()
        else:
            raise EnvironmentError
            
    def set_genespace(self,
                   new_genes):
        
        if not isinstance(new_genes,pd.Index):
            new_genes = pd.Index(new_genes)
        
        sptidx = self._cnt.index
        inter = self.genes.intersection(new_genes)
        
        self.G = new_genes.shape[0]
        self.genes = new_genes
        
        tmp = pd.DataFrame(np.zeros((self.S,self.G)),
                           index = sptidx,
                           columns = new_genes)
        
        tmp[inter] = self._cnt[inter]
        
        self._cnt = tmp
        self._update_identifiers()
        
    def set_metaspace(self,
                   new_cols):
        if self.has_meta:
            if not isinstance(new_cols,pd.Index):
                new_cols = pd.Index(new_cols)
            
            sptidx = self._cnt.index
            inter = self.meta.columns.intersection(new_cols)
            
            self.K = new_cols.shape[0]
            
            tmp = pd.DataFrame(np.zeros((self.S,self.K)),
                               index = sptidx,
                               columns = new_cols)
            
            tmp.iloc[:,:] = np.nan
            tmp = tmp.astype(str)
            tmp[inter] = self._meta[inter]
            
            self._meta = tmp
            self._update_identifiers()
    
    def subset(self,
               idx,
               ax = 0,
               ):
        
        if ax == 0:
            if self.has_meta: self._meta = self._meta.iloc[idx,:]
            self._cnt = self._cnt.iloc[idx,:]
        else:
            if self.has_meta:  self._meta = self._meta.iloc[:,idx]
            self._cnt = self._cnt.iloc[:,idx]
            
        self._update_identifiers()
    
    def idx_of(self,
               feature,
               label,
               ):
        
        if feature in self.meta.columns:
            return np.where(self.meta[feature] == label)[0]
        else:
            return np.array([])
            
    def __call__(self,s):
        
        if np.min(s) > 0 and np.max(s) < self.S:
            if self.has_meta:
                return self._cnt.iloc[s,:], self._meta.iloc[s,:]
            else: 
                return self._cnt.iloc[s,:]
        else:
            raise RuntimeError
            
    def __str__(self,):
        return ' | '.join([f"patient : {self.patient}",
                          f"replicate : {self.replicate}",
                          f"spots : {self.S}",
                          f"genes : {self.G}"])
    
    
    def plot_gene(self,
                  gene,
                     eax = None,
                    ):
        
        if eax:
            ax = eax
            fig = None
        else:
            fig,ax = plt.subplots(1,1, figsize = None)
        
        cval = self._cnt[gene].values
        sf = self._cnt.values.sum(axis = 1).reshape(-1,1)
        sf[sf == 0] = np.nan
        cval = np.divide(cval,sf)
        cval[np.isnan(cval)] = 0.0
        
        if cval.shape[1] > 1:
            cval = cval.sum(axis = 1)
            
        cval = (cval - cval.min()) / (cval.max() - cval.min())
        
        ax.scatter(self.x, self.y,
                   cmap = plt.cm.Blues,
                   vmax = 0.95,
                   vmin = 0.05,
                   c = cval,
                   s = 120,
                   edgecolor = 'black')
        
        if not eax:
            return fig,ax
    
    def plot_custom(self,
                    cvals,
                    figsize = None,
                    mark_feature = None,
                    mark_val = None,
                    eax = None,
                    marker_size = 100,
                    **kwargs,
                    ):
        
        if eax:
            ax = eax
            fig = None
        else:
            fig,ax = plt.subplots(1,1, figsize = None)
        
        if mark_feature:
            idx = self.meta[mark_feature] == mark_val
            edgecolor = np.zeros((self.S,4))
            edgecolor[idx,3] = 0.5
        else:
            edgecolor = None
        
        ax.scatter(self.x,
                   self.y,
                   c = cvals,
                   cmap = plt.cm.Blues,
                   s = marker_size,
                   edgecolor = edgecolor,
                   vmin = kwargs.get('vmin',None),
                   vmax = kwargs.get('vmax',None),
                   )
        
        if not eax:
            return fig, ax
        
    
    def plot_meta(self,
                 feature,
                 var_type = 'continous',
                 marker_size = 120,
                 edgecolor = 'black',
             eax = None,
                    ):
        if self.has_meta:
            if eax:
                ax = eax
                fig = None
            else:
                fig,ax = plt.subplots(1,1, figsize = None)
            
            tvals = self.meta[feature].values.reshape(-1,)
            if var_type == 'continous':
                color = tvals
                if np.sign(tvals.min()) == np.sign(tvals.max()):
                        cmap = plt.cm.Blues
                else:
                    cmap = plt.cm.RdBu
                
                mx = np.max(np.abs(color))
                vmin = -mx
                vmax = mx
                labels = None
            
            else:
                color = np.zeros(tvals.shape)
                cmap = plt.cm.tab20
                labels = []
                for k,cat in enumerate(np.unique(tvals)):
                    color[tvals == cat] = k
                    labels.append(Patch(facecolor = cmap(k), label = cat)) 
                
                vmin = None
                vmax = None
            
            ax.scatter(self.x,self.y,
                       c = color,
                       vmin = vmin,
                       vmax = vmax,
                       cmap = cmap,
                       s = marker_size,
                       label = labels, 
                       edgecolor = edgecolor)
            
            if any(labels):
                ax.legend(handles = labels)
            
            if not eax:
                return fig, ax
        else:
            return None