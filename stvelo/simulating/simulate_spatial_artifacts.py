import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors


def simulate_missegmentation(adata:'AnnData',max_distance_misseg:float=10,max_missegmentation_proportion:float=0.1):
    """Identify the presence of missegmented cells and simulate missegmentation
    Parameters
    ----------
    adata:'AnnData object'
        Adata object including previously simulated cells and spatial positions
    max_distance_misseg:'float'
        Maximum distance that is consider to assume two cells are missegmented. Cells presented at a shorter distance are
        considered to be missegmented
    max_missegmentation_proportion:'float'
        Maximum proportion of the cytoplasmic transcripts of cell1 that can be detected by the missegmented neighboring cell2.

    Returns
    -------
    adata:'AnnData object'
        Adata object containing all cells inputed, whose expression now included the computed missegmentation.
        A new column called 'missegmented cell' is included in obs to identify cells that are considered as missegmented.
    """
    
    
    # we define the closest neighbor to each cell
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(adata.obsm['spatial'])
    distance_mat, neighbours_mat = knn.kneighbors(adata.obsm['spatial'])
    cell_of_interest=neighbours_mat[:,0]
    closest_cell=neighbours_mat[:,1]
    closest_cell_distance=distance_mat[:,1]

    # we filter cells to keep only the missegmented ones
    misseg_cell2=closest_cell[closest_cell_distance<max_distance_misseg]
    misseg_cell1=cell_of_interest[closest_cell_distance<max_distance_misseg]
    misseg_distance=closest_cell_distance[closest_cell_distance<max_distance_misseg]
    
    # we next extract the expression matrix of cells from adata, together with the spliced and unspliced counts
    expression=np.array(adata.X)
    spliced_expression=adata.layers['spliced']
    unspliced_expression=adata.layers['unspliced']
    
    # for every cell we identified as missgmented, we transfer part of the expression between misseg cells
    for index in range(0,len(misseg_cell1)):
        cell1=misseg_cell1[index]
        cell2=misseg_cell2[index]
        added_expression=spliced_expression[cell2,:]*random.uniform(0,max_missegmentation_proportion)
        spliced_expression[cell1,:]=spliced_expression[cell1,:]+added_expression
        spliced_expression[cell2,:]=spliced_expression[cell2,:]-added_expression
        expression[cell1,:]=expression[cell1,:]+added_expression
        expression[cell2,:]=expression[cell2,:]-added_expression

    adata.X=expression
    adata.layers['spliced']=spliced_expression
    adata.layers['unspliced']=unspliced_expression
    # we add a variable in the adata specifying wether the cell is missegmented or not
    adata.obs['missegmented_cell']=list(adata.obs.index.astype(int).isin(misseg_cell1))
    
    return adata

def simulate_cytoplasmic_leakage(adata:'AnnData',max_cytoplasmic_leakage:float=0.1):
    ''' Simulate cytoplasmic counts considered nuclear due to 2D segmentation of a 3D nuclei.
    We consider this effect is systematic in all cells, but due to sampling some genes can be more affected than others
     ----------
    adata:'AnnData object'
        Adata object including previously simulated cells 
    max_cytoplasmic_leakage:float
        Maximum proportion of cytoplasmic counts that can leak into the nucleus counts do 2D segmentation
    Returns
    -------
    adata:'AnnData object'
        Adata object including previously cells where leakage has been simulated
    '''

    expression=np.array(adata.X)
    spliced_expression=adata.layers['spliced']
    unspliced_expression=adata.layers['unspliced']
    allperc=[]
    for index in range(expression.shape[0]):
        perc=[random.uniform(0,max_cytoplasmic_leakage) for e in range(expression.shape[1])]
        allperc.append(np.mean(perc))
        added_expression=spliced_expression[index,:]*perc
        spliced_expression[index,:]=spliced_expression[index,:]-added_expression
        unspliced_expression[index,:]=unspliced_expression[index,:]+added_expression

    adata.X=expression
    adata.layers['spliced']=spliced_expression
    adata.layers['unspliced']=unspliced_expression
    adata.obs['mean_leakage']=allperc
    return adata

def simulate_boundary_underestimation(adata:'AnnData',max_cyto_prop_lost:float=0.1,how:str='random'):
    ''' Simulate an underestimation of the boundaries
     ----------
    adata:'AnnData object'
        Adata object including previously simulated cells 
    max_cyto_prop_lost:float
        Maximum proportion of cytoplasmic counts not properly segmented due do cell boundary underestimation
    how:str
        Method to simulate boundary underestimation. It can be 'random','with_descending_trajectory' or 'with ascending trajectory'
    Returns
    -------
    adata:'AnnData object'
        Adata object including previously cells where cell boundary underestimation has been considered
    '''

    expression=np.array(adata.X)
    spliced_expression=adata.layers['spliced']
    unspliced_expression=adata.layers['unspliced']
    allperc=[]
    if how=='random':
        for index in range(expression.shape[0]):
            perc=[random.uniform(0,max_cyto_prop_lost) for e in range(expression.shape[1])]
            allperc.append(np.mean(perc))
            lost_expression=spliced_expression[index,:]*perc
            spliced_expression[index,:]=spliced_expression[index,:]-lost_expression
            
    if how=='with_descending_trajectory':
        dependent_max=a.obs['true_t'].div(a.obs['true_t'].max())*max_cyto_prop_lost
        for index in range(expression.shape[0]):
            perc=[random.uniform(0,dependent_max[index]) for e in range(expression.shape[1])]
            allperc.append(np.mean(perc))
            lost_expression=spliced_expression[index,:]*perc
            spliced_expression[index,:]=spliced_expression[index,:]-lost_expression
    if how=='with_ascending_trajectory':
        dependent_max=(1-(a.obs['true_t'].div(a.obs['true_t'].max())))*max_cyto_prop_lost
        for index in range(expression.shape[0]):
            perc=[random.uniform(0,dependent_max[index]) for e in range(expression.shape[1])]
            allperc.append(np.mean(perc))
            lost_expression=spliced_expression[index,:]*perc
            spliced_expression[index,:]=spliced_expression[index,:]-lost_expression

    adata.X=spliced_expression+unspliced_expression
    adata.layers['spliced']=spliced_expression
    adata.layers['unspliced']=unspliced_expression
    adata.obs['boundary_underestimation']=allperc
    return adata


