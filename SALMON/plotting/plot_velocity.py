import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_unspliced_vs_spliced(a,genes:list=a.var.index,color:str=None,spotsize:int=2):
    """ Plot, for the selected list of genes, the unspliced vs spliced transcripts
    
    Parameters:
    a (AnnData): AnnData object containing spliced ans unspliced counts in adata.layers['spliced'] and adata.layers['unspliced']
    genes(list): list of genes to visualize
    color(str): numeric variable in adata.obs to use to color the spots. If None, color is the same for all spots
   
    Returns:
    None
    """
    exp=a.to_df()
    s=np.array(a.layers['spliced'])
    u=np.array(a.layers['unspliced'])
    fig,ax=plt.subplots(1,len(genes),figsize=(len(genes)*3,3),sharey=True)
    num=0
    for e in genes:
        if color is not  None: 
            ax[num].scatter(s[:,int(e)],u[:,int(e)],s=spotsize,c=a.obs[color])
        else:
            ax[num].scatter(s[:,int(e)],u[:,int(e)],s=spotsize)
        ax[num].set_title(e)
        if num==0:
                ax[num].set_ylabel('unspliced')
        ax[num].set_xlabel('spliced')    
        num=num+1
        
        
        
def plot_expression_vs_true_time(a,genes:list=a.var.index,color:str=None,spotsize:int=2):
    """ Plot, for the selected list of genes, the expression vs true time
    
    Parameters:
    a (AnnData): AnnData object containing the true_time and counts in a.obs['true_t'] and the expresison matrix respectively.
    genes(list): list of genes to visualize
    color(str): numeric variable in adata.obs to use to color the spots. If None, color is the same for all spots
   
    Returns:
    None
    """
    exp=a.to_df()
    fig,ax=plt.subplots(1,len(genes),figsize=(len(genes)*3,3),sharey=True)
    num=0
    for e in genes:
        if color is not  None: 
            ax[num].scatter(a.obs['true_t'],exp[e],s=spotsize,c=a.obs['true_t'])
        else:
            ax[num].scatter(a.obs['true_t'],exp[e],s=spotsize)
        ax[num].set_title(e)
        if num==0:
                ax[num].set_ylabel(e + 'Expression')
        ax[num].set_xlabel('True time')  
        num=num+1
        
        
def plot_velocity_vs_true_t(a,genes:list=a.var.index,color:str=None,spotsize:int=2):
    """Plot, in simulations, the velocity of a gene versus its true velocity
    
    Parameters:
    a (AnnData): AnnData object containing the the true velocity in obs and the computed velocity in a.layers['velocity']
    genes(list): list of genes to visualize
    color(str): numeric variable in adata.obs to use to color the spots. If None, color is the same for all spots
   
    Returns:
    None
    """        
 
    exp=a.to_df()
    velo=pd.DataFrame(np.array(a.layers['velocity']),columns=a.var.index)
    
    fig,ax=plt.subplots(1,len(genes),figsize=(len(genes)*3,3),sharey=True)
    num=0
    for e in genes:
        if color is not  None: 
            ax[num].scatter(a.obs['true_t'],velo.loc[:,e],s=spotsize,c=a.obs['true_t'])
        else:
            ax[num].scatter(a.obs['true_t'],velo.loc[:,e],s=spotsize)
        ax[num].set_title(e)
        if num==0:
                ax[num].set_ylabel('velocity')
        ax[num].set_xlabel('True time')  
        num=num+1
        
        