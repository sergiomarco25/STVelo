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
from tqdm import tqdm




def define_cell_polarities_angles(adatafilt:AnnData):
    """ DEPRECATED FUNCTION. Function to calculate the polarity based on angles 
   
    Parameters:
    adatafilt (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with computed polarity
    """
    cells=adatafilt.uns['spots']['cell_id'].unique()
    feats=adatafilt.uns['spots']['feature_name'].unique()
    positiondict=dict(zip(list(feats),range(0,len(feats))))
    resarray=np.zeros([len(cells),len(feats)])
    resx=np.zeros([len(cells),len(feats)])
    resy=np.zeros([len(cells),len(feats)])
    id2x2=dict(zip(adatafilt.obs['cell_id'],adatafilt.obs['x_centroid']))
    id2y2=dict(zip(adatafilt.obs['cell_id'],adatafilt.obs['y_centroid']))
    ee=0
    cell_ids=[]
    for a,g in tqdm(adatafilt.uns['spots'].groupby('cell_id')):
        xcell=id2x2[a]
        ycell=id2y2[a]
        ii=0
        meang=g.groupby('feature_name').mean()
        meang['polarity']=np.sqrt((meang['x_location']-xcell)**2+(meang['y_location']-ycell)**2)
        #dici=dict(zip(meang.index,meang['nucleus_distance']))
        resarray[ee,list(meang.index.map(positiondict))]=meang['polarity']
        resx[ee,list(meang.index.map(positiondict))]=meang['x_location']-xcell
        resy[ee,list(meang.index.map(positiondict))]=meang['y_location']-ycell
        ee=ee+1
        cell_ids.append(a)
    polarity=pd.DataFrame(resarray,index=cell_ids,columns=feats)
    xgene=pd.DataFrame(resx,index=cell_ids,columns=feats)
    ygene=pd.DataFrame(resy,index=cell_ids,columns=feats)
    adatafilt.obsm['polarity']=polarity.loc[adatafilt.obs['cell_id'],:]
    adatafilt.obsm['polarity']=adatafilt.obsm['polarity'].loc[:,adatafilt.var['gene_id']]
    adatafilt.obsm['xgene']=xgene.loc[adatafilt.obs['cell_id'],:]
    adatafilt.obsm['xgene']=adatafilt.obsm['xgene'].loc[:,adatafilt.var['gene_id']]
    adatafilt.obsm['ygene']=ygene.loc[adatafilt.obs['cell_id'],:]
    adatafilt.obsm['ygene']=adatafilt.obsm['ygene'].loc[:,adatafilt.var['gene_id']]
    adatafilt.obsm['polarity'][adatafilt.obsm['polarity']==0]=np.nan
    return adatafilt