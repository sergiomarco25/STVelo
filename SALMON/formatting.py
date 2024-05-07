import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import gzip
import shutil
import os.path
from scipy.io import mmread
import tifffile as tf
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from anndata import AnnData
import json


def filter_distant_reads(adata,max_distance=10):
    ''' Function to filter the transcripts assign to cells based on the distance of reads to nucleus''' 
    adata.uns['spots']=adata.uns['spots'][adata.uns['spots']['nucleus_distance']<max_distance]
    return adata

def transcript_based_adata(adata):
    '''Function to recompute adata counts, stored in adata.X, based on the read-based info, in adata.uns['spots']'''
    ct1=pd.crosstab(adata.uns['spots']['cell_id'],adata.uns['spots']['feature_name'])
    adataobs=adata.obs.loc[adata.obs['cell_id'].isin(ct1.index),:]
    av=adata.var['gene_id'][adata.var['gene_id'].isin(ct1.columns)]#.isin(adata1.var.index)
    av2=adata.var[adata.var['gene_id'].isin(ct1.columns)]#.isin(adata1.var.index)
    ct1=ct1.loc[:,av]
    adataobs.index=adataobs['cell_id']
    adataobs.index.name='ind'
    ct1=ct1.loc[ct1.index.isin(adataobs['cell_id']),:]
    ct1=ct1.loc[adataobs['cell_id'],:]
    adata1nuc=sc.AnnData(np.array(ct1),obs=adataobs)
    adata1nuc.var=av2
    adata1nuc.uns['spots']=adata.uns['spots']
    return adata1nuc

def adata_based_transcripts(adata):
    adata.uns['spots']=adata.uns['spots'][adata.uns['spots']['cell_id'].astype(str).isin(adata.obs['cell_id'].astype(str))]
    return adata
