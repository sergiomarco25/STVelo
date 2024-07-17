import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors



def simulate_static_population(n_transcripts:int=50,n_cells:int=1200,size_x:int=500,size_y:int=500):
    """ Function to simulate a static populations from invariant expression in your dataset.
    Parameters
    ----------
    n_transcripts:'AnnData object'
        Number of transcripts
    n_cells:'int'
        Number of cells to be simulated
    size_x:'float'
        X limit in the space of your cohabiting population
    size_y:'float'
        Y limit in the space of your cohabiting population

    Returns
    -------
    a2:'AnnData object'
        Adata object containing the simulated expression of the cohabining cells
    """
    a2=scv.datasets.simulation(n_obs=n_cells,n_vars=n_transcripts) [round(n_transcripts/2),:]
    indis=list(a2.var.index)
    random.shuffle(indis)
    a2=a2[:,indis]
    a2.var=a2.var.reset_index()
    ### simulate unspliced
    sim_array_u=np.zeros([a2.X[0,:].shape[0],n_cells])
    ii=0
    for e in a2.X[0,:]:
        sim_array_u[ii,:]=np.random.poisson(e/2,size=n_cells)
        ii=ii+1
    sim_array_s=np.zeros([a2.X[0,:].shape[0],n_cells])
    ii=0
    ### simulate spliced
    for e in a2.X[0,:]:
        sim_array_s[ii,:]=np.random.poisson(e/2,size=n_cells)
        ii=ii+1
    sim_array=sim_array_u.transpose()+sim_array_s.transpose()
    # create anndata object
    a2=sc.AnnData(sim_array)
    a2.layers['spliced']=sim_array_s.transpose()
    a2.layers['unspliced']=sim_array_u.transpose()
    a2.obs['true_t']=np.nan
    a2.obs=a2.obs.reset_index()
    a2=simulate_space(a2,how='random',size_x=size_x,size_y=size_y)
    a2.obs['kind']='cohabiting_population'
    return a2
