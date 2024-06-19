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


def define_cell_polarities(adatafilt:AnnData):
    """ Define cell polarities based on data present on adata.uns['spots']. Polarity in here is defined as the mean distance of the gene specific centroids to the cellular centroid
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with mean and maximum polarity included in adata.obs
    
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
        g=g.loc[:,['x_location','y_location','feature_name']]
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
    adatafilt.obs['mean_polarity']=np.mean(adatafilt.obsm['polarity'],axis=1)
    adatafilt.obs['max_polarity']=np.max(adatafilt.obsm['polarity'],axis=1)
    return adatafilt


def extract_nuclear_and_cytoplasmic(adata:AnnData):
    """ Function to compute nuclear expression matrix and cytoplasmic expression matrix
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adata : Cell expression in AnnData format with redefined adata.uns['spots'], including nuclear expression matrix in adata.uns['nuclear_expression'] and cytoplasmic expression in adata.uns['cytoplasmic_expression']
    
   """

    didi={0:'cyt',1:'nuc'}
    adata.uns['spots']['feature_n_loc']=adata.uns['spots']['feature_name']+'_'+adata.uns['spots']['overlaps_nucleus'].map(didi)
    div_exp=pd.crosstab(adata.uns['spots']['cell_id'],adata.uns['spots']['feature_n_loc'])
    adata=adata[adata.obs['cell_id'].isin(div_exp.index),:]
    #adata.obs.index.name='ind'
    div_exp=div_exp.loc[div_exp.index.isin(adata.obs['cell_id']),:]
    div_exp=div_exp.loc[adata.obs['cell_id'],:]
    nuc_cols=[e for e in div_exp.columns if '_nuc' in e]
    cyt_cols=[e for e in div_exp.columns if '_cyt' in e]
    adata.uns['nuclear_expression']=div_exp.loc[:,nuc_cols]
    adata.uns['cytoplasmic_expression']=div_exp.loc[:,cyt_cols]
    adata.uns['nuclear_expression'].columns=[c.replace('_nuc','') for c in adata.uns['nuclear_expression'].columns]
    adata.uns['cytoplasmic_expression'].columns=[c.replace('_cyt','') for c in adata.uns['cytoplasmic_expression'].columns]
    return adata

def nuclear_and_cytoplasmic_characteristics(adata:AnnData,minimum_expression=0):
    """ Function to extract the main characteristics of the nuclear and cytoplasmic expression
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']
    minimum_expression (int): minimum expression of a gene in a cell to be considered expressed in each compartment

    Returns:
    adata : Cell expression in AnnData format with with main compartment characteristics defined
    
   """
    adata=extract_nuclear_and_cytoplasmic(adata)
    adata.obs['nuc_and_cyt_genes']=list(np.sum(((adata.uns['cytoplasmic_expression']>minimum_expression)*1+(adata.uns['nuclear_expression']>minimum_expression)*1)==2,axis=1))
    adata.obs['cyt_genes']=list(np.sum(((adata.uns['cytoplasmic_expression']>minimum_expression)*1+(adata.uns['nuclear_expression']<=minimum_expression)*1)==2,axis=1))
    adata.obs['nuc_genes']=list(np.sum(((adata.uns['cytoplasmic_expression']<=minimum_expression)*1+(adata.uns['nuclear_expression']>minimum_expression)*1)==2,axis=1))
    adata.obs['expressed_genes']=adata.obs['nuc_genes']+adata.obs['cyt_genes']+adata.obs['nuc_and_cyt_genes']
    adata.obs['nuc_and_cyt_genes_proportion']=adata.obs['nuc_and_cyt_genes']/adata.obs['expressed_genes']
    adata.obs['cyt_genes_proportion']=adata.obs['cyt_genes']/adata.obs['expressed_genes']
    adata.obs['nuc_genes_proportion']=adata.obs['nuc_genes']/adata.obs['expressed_genes']
    adata.obs['cyt_counts']=list(np.sum((adata.uns['cytoplasmic_expression']),axis=1))
    adata.obs['nuc_counts']=list(np.sum((adata.uns['nuclear_expression']),axis=1))
    adata.obs['cyt_counts_proportion']=adata.obs['cyt_counts']/(adata.obs['cyt_counts']+adata.obs['nuc_counts'])
    adata.obs['nuc_counts_proportion']=adata.obs['nuc_counts']/(adata.obs['cyt_counts']+adata.obs['nuc_counts'])
    return adata



from sklearn.decomposition import PCA
def polarization_based_pca(adatafilt:AnnData,plot=False,min_gene_counts:int=0):

    """ Function to compute polarization based on the relative importance of PC1 in the coordinates obtained from gene_specific centroids
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']
    plot(boolean): whether to plot associated plots or not. 
    min_gene_counts(int): minimum number of counts per cell for each gene to consider that gene in the ANALYSIS.
    
    Returns:
    adata : Cell expression in AnnData format with polarization scores based on the importance of PCA
    
   """
    cells=adatafilt.uns['spots']['cell_id'].astype(str).unique()
    feats=adatafilt.uns['spots']['feature_name'].unique()
    positiondict=dict(zip(list(feats),range(0,len(feats))))
    resarray=np.zeros([len(cells),len(feats)])
    resx=np.zeros([len(cells),len(feats)])
    resy=np.zeros([len(cells),len(feats)])
    id2x2=dict(zip(adatafilt.obs['cell_id'].astype('category').astype(str),adatafilt.obs['x_centroid']))
    id2y2=dict(zip(adatafilt.obs['cell_id'].astype('category').astype(str),adatafilt.obs['y_centroid']))
    ee=0
    cell_ids=[]
    variances=[]
    centroid_gene_diff=[]
    asall=[]
    for a,g in tqdm(adatafilt.uns['spots'].groupby('cell_id')):
        try:
            xcell=id2x2[str(a)]
            ycell=id2y2[str(a)]
            ii=0
            g=g.loc[:,['feature_name','y_location','x_location']]
            gco=g.groupby('feature_name').count()
            g=g[g['feature_name'].isin(gco.index[gco.iloc[:,0]>min_gene_counts-1])]
            meang=g.groupby('feature_name').mean()
            ex=pd.DataFrame(np.array([meang['x_location']-xcell,meang['y_location']-ycell])).transpose()
            ex.index=meang.index
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(ex)
            variance=pca.explained_variance_ratio_[0]
            if plot==True:
                if variance>0.95:
                    print(variance)
                    print(a)
                    # plots
                    plt.figure()
                    plt.scatter(ex.iloc[:,0],ex.iloc[:,1])
                    plt.hlines(0,xmin=-5,xmax=5,color='red')
                    plt.vlines(0,ymin=-5,ymax=5,color='red')
                    plt.show()

            variances.append(variance)
            centroid_gene_diff.append(np.mean(np.sqrt(np.sum(ex**2,axis=1))))
            asall.append(str(a))
        except:
            variances.append(np.nan)
            centroid_gene_diff.append(np.nan)
            asall.append(a)
    id2var=dict(zip(asall,variances))
    id2centroidgenediff=dict(zip(asall,centroid_gene_diff))
    adatafilt.obs['polarity_pc']=adatafilt.obs['cell_id'].astype(str).map(id2var)
    adatafilt.obs['mean_distance_to_gene_centroids']=adatafilt.obs['cell_id'].astype(str).map(id2centroidgenediff)
    return adatafilt


def compute_nuclear_centroid(adatafilt:AnnData):
    """ Function to calculate the centroid of the nuclei based on nuclei-assigned ranscripts
   
    Parameters:
    adata (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with nuclear centroid position
    
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
    nuclear_centroid_x=[]
    nuclear_centroid_y=[]
    centroid_to_nucentroid=[]
    asall=[]
    for a,g in tqdm(adatafilt.uns['spots'].groupby('cell_id')):
            try:
                gnuc=g[g['overlaps_nucleus']==1]
                nuclear_centroid_x.append(gnuc['x_location'].mean())
                nuclear_centroid_y.append(gnuc['y_location'].mean())  
                asall.append(a)
                xcell=id2x2[a]
                ycell=id2y2[a]
                centroid_to_nucentroid.append(np.sqrt(((gnuc['x_location'].mean()-xcell)**2)+(gnuc['y_location'].mean()-ycell)**2))
            except:
                asall.append(a)
                nuclear_centroid_x.append(np.nan)
                nuclear_centroid_y.append(np.nan)  
                centroid_to_nucentroid.append(np.nan)
    id2nuc_x=dict(zip(asall,nuclear_centroid_x))
    id2nuc_y=dict(zip(asall,nuclear_centroid_y))
    id2ctonuc=dict(zip(asall,centroid_to_nucentroid))
    adatafilt.obs['nuclei_centroid_x']=adatafilt.obs['cell_id'].map(id2nuc_x)
    adatafilt.obs['nuclei_centroid_y']=adatafilt.obs['cell_id'].map(id2nuc_y)
    adatafilt.obs['distance_centroid_to_nuccentroid']=adatafilt.obs['cell_id'].map(id2ctonuc)
    return adatafilt






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


def centrality_scores(adatafilt:AnnData):


    """ Calculate centrality scores for genes using squidpy's implementation of these scores. Includes:
        - closeness centrality: how close a group is to other nodes
        - degree centrality: fraction of connected non-group members
        - clustering coeffeicient: measure of the degree to which nodes cluster
   
    Parameters:
    adatafilt (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with centrality scores computed per gene and stored in adatafilt.uns['{'ac/cc/dc'}_score']
    
    """

    cells=adatafilt.uns['spots']['cell_id'].unique()
    feats=adatafilt.uns['spots']['feature_name'].unique()
    positiondict=dict(zip(list(feats),range(0,len(feats))))
    resarray=np.zeros([len(cells),len(feats)])
    resx=np.zeros([len(cells),len(feats)])
    resy=np.zeros([len(cells),len(feats)])
    ee=0
    cell_ids=[]
    dc_all=pd.DataFrame(index=adatafilt.obs.index,columns=adatafilt.var.index)
    ac_all=pd.DataFrame(index=adatafilt.obs.index,columns=adatafilt.var.index)
    cc_all=pd.DataFrame(index=adatafilt.obs.index.unique(),columns=adatafilt.var.index)
    adatafilt.uns['spots']['feature_name']=adatafilt.uns['spots']['feature_name'].astype(str)
    import warnings
    warnings.filterwarnings("ignore")
    from copy import deepcopy
    import time
    for a,g in tqdm(adatafilt.uns['spots'].groupby('cell_id')):
        try:
        # do stuff
            celldata=sc.AnnData(obs=g)
            celldata.obs['feature_name']=celldata.obs['feature_name'].astype('category')
            celldata.obsm['spatial']=np.array(celldata.obs.loc[:,['x_location','y_location']])
            sq.gr.spatial_neighbors(celldata,n_neighs=10)
            sq.gr.centrality_scores(celldata, "feature_name")
            #df_central.index = meta_leiden.index.tolist()
            dc_all.loc[a,df_central.index]=celldata.uns['feature_name_centrality_scores']['degree_centrality']
            cc_all.loc[a,df_central.index]=celldata.uns['feature_name_centrality_scores']['closeness_centrality']
            ac_all.loc[a,df_central.index]=celldata.uns['feature_name_centrality_scores']['average_clustering']
        except Exception: 
            pass
    adatafilt.uns['cc_score']=cc_all
    adatafilt.uns['ac_score']=ac_all
    adatafilt.uns['dc_score']=dc_all
    
    return adatafilt


def calcualte_densities(adatafilt:AnnData):
    """ Compute read density in cell, cytoplasm and nuclei
    Note: that it requires counts by compartment and areas
   
    Parameters:
    adatafilt (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with cell,nuclear and cytoplasic densities computed
    
   """

    adatafilt.obs['cytoplasm_area']=adatafilt.obs['cell_area']-adatafilt.obs['nucleus_area']
    adatafilt.obs['nucleus_area_proportion']=adatafilt.obs['nucleus_area']/adatafilt.obs['cell_area']
    adatafilt.obs['cyt_density']=adatafilt.obs['cyt_counts']/adatafilt.obs['cytoplasm_area']
    adatafilt.obs['nuc_density']=adatafilt.obs['nuc_counts']/adatafilt.obs['nucleus_area']
    adatafilt.obs['cell_density']=(adatafilt.obs['cyt_counts']+adatafilt.obs['nuc_counts'])/adatafilt.obs['cell_area']
    return adatafilt


def nuclear_to_cytoplasmic_correlation(adatafilt:AnnData):
    """ Compute nuclear-to_cytoplasmic correlation for each cell
    Note: that it requires counts by compartment (nuc vs cyt)
   
    Parameters:
    adatafilt (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with nuclear_to_cytoplasmic correlation computed for each cell
    
   """ 
    
    nuc_cyt_corr=[]
    adatafilt.uns['nuclear_expression']=adatafilt.uns['nuclear_expression'].loc[:,adatafilt.uns['nuclear_expression'].columns.isin(adatafilt.uns['cytoplasmic_expression'].columns)]
    adatafilt.uns['cytoplasmic_expression']=adatafilt.uns['cytoplasmic_expression'].loc[:,adatafilt.uns['cytoplasmic_expression'].columns.isin(adatafilt.uns['nuclear_expression'].columns)]
    for c in tqdm(adatafilt.obs.index):#adatafilt.uns['nuclear_expression'].index):
        nuc_cyt_corr.append(np.corrcoef(adatafilt.uns['nuclear_expression'].loc[c,:],adatafilt.uns['cytoplasmic_expression'].loc[c,:])[0,1])
    adatafilt.obs['cyt_nuc_correlation']=nuc_cyt_corr
    return adatafilt

def gene_nuclear_to_cytoplasmic_correlation(adatafilt:AnnData):
     
     
     """ Compute, for each gene, the nuclear-to_cytoplasmic correlation across cells
    Note: that it requires counts by compartment
   
    Parameters:
    adatafilt (AnnData): Cell expression in AnnData format, including read information in adata.uns['spots']

    Returns:
    adatafilt : Cell expression in AnnData format with nuclear_to_cytoplasmic correlation computed for each gene
    
     """ 
     nuc_cyt_corr=[]
     for c in tqdm(adatafilt.var.index):#adatafilt.uns['nuclear_expression'].index):

        try:
            nuc_cyt_corr.append(np.corrcoef(adatafilt.uns['nuclear_expression'].loc[:,c],adatafilt.uns['cytoplasmic_expression'].loc[:,c])[0,1])
        except:
            nuc_cyt_corr.append(np.nan)
     adatafilt.var['nuc_cyt_correlation']=nuc_cyt_corr
     return adatafilt
