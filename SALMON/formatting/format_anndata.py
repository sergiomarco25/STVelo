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
    """ Filter reads that are too far from the cells from adata.uns['spots']
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']
    max_distance(float): Maxium distance from each read to their closest nucleus edge required to be kept in the analyis

    Returns:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots'] with filtered reads

   """
    adata.uns['spots']=adata.uns['spots'][adata.uns['spots']['nucleus_distance']<max_distance]
    return adata

def transcript_based_adata(adata):
    """ Function to recompute adata counts, stored in adata.X, based on the read-based info, in adata.uns['spots']
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adata1nuc : Cell expression in AnnData format redefined based on the read information in adata.uns['spots'] 
   """
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
    """ Function to filter in adata.uns['spots'] based on the cells present in the AnnData.obs info (in cell_id)
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adata : Cell expression in AnnData format with redefined adata.uns['spots'] 
    
   """
    
    adata.uns['spots']=adata.uns['spots'][adata.uns['spots']['cell_id'].astype(str).isin(adata.obs['cell_id'].astype(str))]
    return adata



def extract_nuclear_and_cytoplasmic(adata:AnnData):
    """ Function to compute nuclear expression matrix and cytoplasmic expression matrix

    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adata : Cell expression in AnnData format with redefined adata.uns['spots'], including nuclear expression matrix in     
    adata.uns['nuclear_expression'] and cytoplasmic expression in adata.uns['cytoplasmic_expression']

    """

    didi={0:'cyt',1:'nuc'}
    adata.uns['spots']['feature_n_loc']=adata.uns['spots']['feature_name']+'_'+adata.uns['spots']['overlaps_nucleus'].map(didi)
    div_exp=pd.crosstab(adata.uns['spots']['cell_id'],adata.uns['spots']['feature_n_loc'])
    adata=adata[adata.obs['cell_id'].isin(div_exp.index),:]
    div_exp=div_exp.loc[div_exp.index.isin(adata.obs['cell_id']),:]
    div_exp=div_exp.loc[adata.obs['cell_id'],:]
    nuc_cols=[e for e in div_exp.columns if '_nuc' in e]
    cyt_cols=[e for e in div_exp.columns if '_cyt' in e]
    adata.uns['nuclear_expression']=div_exp.loc[:,nuc_cols]
    adata.uns['cytoplasmic_expression']=div_exp.loc[:,cyt_cols]
    adata.uns['nuclear_expression'].columns=[c.replace('_nuc','') for c in adata.uns['nuclear_expression'].columns]
    adata.uns['cytoplasmic_expression'].columns=[c.replace('_cyt','') for c in adata.uns['cytoplasmic_expression'].columns]
    return adata


