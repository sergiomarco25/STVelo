import scanpy as sc
import scvelo as scv
from typing import Optional
from anndata import AnnData

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:
    def __init__(self, adata, config):
        self.adata = adata
        self.config = config

    def preprocess_data(self):
        params = self.config.get('preprocess_params', {})
        functions = self.config.get('functions_to_apply', {})

        if functions.get('filter_cells', False):
            min_counts = params.get('min_counts', None)
            if min_counts is not None:
                sc.pp.filter_cells(self.adata, min_counts=min_counts)

        if functions.get('filter_genes', False):
            min_cells = params.get('min_cells', None)
            if min_cells is not None:
                sc.pp.filter_genes(self.adata, min_cells=min_cells)

        if functions.get('normalize_total', False):
            sc.pp.normalize_total(self.adata)

        if functions.get('log1p', False):
            sc.pp.log1p(self.adata)

        if functions.get('pca', False):
            sc.pp.pca(self.adata)

        if functions.get('neighbors', False):
            n_neighbors = params.get('n_neighbors', 10)
            n_pcs = params.get('n_pcs', 50)
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        if functions.get('umap', False):
            min_dist = params.get('min_dist', 0.5)
            sc.tl.umap(self.adata, min_dist=min_dist)

        if functions.get('leiden', False):
            resolution = params.get('resolution', 1.0)
            sc.tl.leiden(self.adata, resolution=resolution)

        if functions.get('moments', False):
            n_pcs = params.get('n_pcs', None)
            n_neighbors = params.get('n_neighbors', None)
            scv.pp.moments(self.adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

        return self.adata
    
def preprocess_data_velovi(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata

