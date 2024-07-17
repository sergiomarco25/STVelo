import random
from sklearn.neighbors import NearestNeighbors

def simulate_space(adata:'AnnData',how:str='random',size_x:str=1000,size_y:str=1000,spread:int=10):
    '''Simulate the spatial distribution of cells
    
    Parameters
    ----------
    adata:'AnnData object'
        Adata object including previously simulated cells 
    how:'str'
        Method to use to simulate the space.Options are:
            - 'random': assign a random position to each cell
            - 'spatial_gradient': cells are positioned forming a gradient in y according to their diff. trajectory, included in (y).
    size_x:'float':
        X dimension of the tissue space simulated 
    size_y:'float':
        Y dimension of the tissue space simulated
    spread:'int':
        If 'spatial_gradient' is considered as a method, spreads indicates how many units can the Y coordinate of a cell differ the Y position predicted from 'true_t'
        The higher this value is, the less clear the gradient will be

    Returns
    -------
    adata:'AnnData object'
        Adata object including previously simulated cells with spatial position 
    '''
    if how=='random':
        n_cells=adata.obs.shape[0]
        xpos=[random.uniform(0,size_x) for _ in range(n_cells)]
        ypos=[random.uniform(0,size_y) for _ in range(n_cells)]
    if how=='spatial_gradient':
        n_cells=adata.obs.shape[0]
        xpos=[random.uniform(0,size_x) for _ in range(n_cells)]
        n_cells=adata.obs.shape[0]
        ypos=[random.uniform(0,size_y) for _ in range(n_cells)]
        relative_y=a.obs['true_t'].div(a.obs['true_t'].max())*size_y
        ypos=[i+random.uniform(-spread,spread) for i in relative_y]

    adata.obs['x']=xpos
    adata.obs['y']=ypos
    adata.obsm['spatial']=np.array([list(xpos),list(ypos)]).transpose()
    return adata

