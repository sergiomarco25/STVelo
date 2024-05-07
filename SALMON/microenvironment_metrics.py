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
import squidpy as sq
from tqdm import tqdm

# new binning function using neighbors
# in here, condit should be the sample name
# sname is the name of your cluster column (aka leiden for example)
def format_data_neighs_radius(adata,sname,condit,radius=20):
    adata_copy_int=adata
    sq.gr.spatial_neighbors(adata_copy_int,radius=radius,coord_type = 'generic')
    result=np.zeros([adata.shape[0],len(adata_copy_int.obs[sname].unique())])
    n=0
    tr=adata_copy_int.obsp['spatial_distances'].transpose()
    tr2=tr>0
    from tqdm import tqdm
    for g in tqdm(adata_copy_int.obs[sname].unique()):
        epv=adata_copy_int.obs[sname]==g*1
        opv=list(epv*1)
        result[:,n]=tr2.dot(opv)
        n=n+1
    expmat=pd.DataFrame(result,columns=adata_copy_int.obs[sname].unique())
    adata1=sc.AnnData(expmat,obs=adata.obs)
    adata1.obs['sample']=condit
    adata1.obs['condition']=condit
    adata.obs['neighborhood_diversity']=np.sum(adata1.to_df()>0,axis=1)
    adata.obs['neighborhood_density']=np.sum(adata1.to_df(),axis=1)
    return adata

def compute_closest_neighbor_distance(adatafilt):
    adata_copy_int=adatafilt
    sq.gr.spatial_neighbors(adata_copy_int,n_neighs=1,coord_type = 'generic')
    n=0
    tr=adata_copy_int.obsp['spatial_distances'].transpose()
    tr2=tr>0
    knndist=tr2.dot(adata_copy_int.obsm['spatial'])
    adatafilt.obs['closest_cell_distance']=np.sqrt(np.sum((adata_copy_int.obsm['spatial']-knndist)**2,axis=1))
    return adatafilt