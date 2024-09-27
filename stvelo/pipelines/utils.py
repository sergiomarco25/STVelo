import pandas as pd
from anndata import AnnData

def keep_common_cells(adata_dict: dict) -> dict:
    """
    Keep only the common cells across multiple AnnData objects in a dictionary.
    
    This function identifies the set of cells (obs indices) that are present in all the AnnData objects
    in the given dictionary and filters each AnnData object to retain only those common cells.

    Parameters:
    ----------
    adata_dict : dict
        A dictionary where keys are names (str) and values are AnnData objects. 
        Each AnnData object must have a corresponding `obs` dataframe with cell indices.

    Returns:
    -------
    adata_dict : dict
        The updated dictionary where each AnnData object contains only the common cells across all the AnnData objects.

    Example:
    --------
    ```python
    filtered_dict = keep_common_cells(adata_dict)
    ```

    Notes:
    ------
    - The function modifies the original AnnData objects in the input dictionary.
    - Cells that do not exist in all AnnData objects are removed.
    """
    # Initialize the result with the intersection of obs indices across all AnnData objects
    result = None

    # Identify the common set of cells across all AnnData objects
    for k, adata in adata_dict.items():
        try:
            result.intersection_update(adata.obs.index)
        except:
            result = set(adata.obs.index)

    # Filter each AnnData object to retain only the common cells
    for k, adata in adata_dict.items():
        adata_dict[k] = adata[adata.obs.index.isin(result), :]

    return adata_dict
