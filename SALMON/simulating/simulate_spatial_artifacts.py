import random
from sklearn.neighbors import NearestNeighbors



def compute_missegmentation(adata:'AnnData',max_distance_misseg:float=10,max_missegmentation_proportion:float=0.1):
    """Identify the presence of missegmented cells and compute missegmentation
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