import scvelo as scv
import scanpy as sc
import os 
from pathlib import Path
import numpy as np 
import pandas as pd 
from anndata import AnnData
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
from scvelo.plotting.simulation import compute_dynamics
from sklearn.metrics import accuracy_score


def get_fit_scvelo(adata):
    """
    Compute spliced and unspliced RNA velocity fits for each gene in an AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix of shape `n_obs` x `n_vars` (cells x genes), 
        typically used in scRNA-seq analysis. It must contain precomputed 
        layers "Ms" (spliced) and "Mu" (unspliced) matrices, representing 
        the spliced and unspliced transcript counts.

    Returns:
    --------
    scvelo_fit_s : pd.DataFrame
        DataFrame containing the computed spliced fits for each gene 
        across cells. Rows represent cells, and columns represent genes.
    
    scvelo_fit_u : pd.DataFrame
        DataFrame containing the computed unspliced fits for each gene 
        across cells. Rows represent cells, and columns represent genes.

    Description:
    ------------
    The function iterates through the genes in the `adata` object and computes 
    the RNA velocity fits (both spliced and unspliced) for each gene using the 
    `compute_dynamics` function. It stores the results in two DataFrames: one 
    for spliced RNA counts and one for unspliced RNA counts.
    """
    # Initialize DataFrames with zeros, having the same shape as "Ms" and "Mu" layers.
    scvelo_fit_s = pd.DataFrame(
        np.zeros_like(adata.layers["Ms"]),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    scvelo_fit_u = pd.DataFrame(
        np.zeros_like(adata.layers["Mu"]),
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # Iterate through each gene, compute its RNA velocity dynamics, and store the fits.
    for i, g in tqdm(enumerate(adata.var_names)):
        _, ut, st = compute_dynamics(
            adata,
            basis=adata.var_names[i],
            key="fit",
            extrapolate=False,
            sort=False,
            t=None,
        )
        # Store spliced and unspliced fits in the respective DataFrames
        scvelo_fit_s[g] = st
        scvelo_fit_u[g] = ut
    
    return scvelo_fit_s, scvelo_fit_u

def compute_mse(ms, mu, mn, mc, fit_s, fit_u, fit_n, fit_c):
    """
    Compute the Mean Squared Error (MSE) between actual and fitted values 
    for different RNA layers (spliced, unspliced, nuclear, and cytoplasmic) 
    and return the log10 MSE ratio.

    Parameters:
    -----------
    ms : np.ndarray
        Observed spliced RNA values.
    mu : np.ndarray
        Observed unspliced RNA values.
    mn : np.ndarray
        Observed nuclear RNA values.
    mc : np.ndarray
        Observed cytoplasmic RNA values.
    fit_s : np.ndarray
        Fitted spliced RNA values.
    fit_u : np.ndarray
        Fitted unspliced RNA values.
    fit_n : np.ndarray
        Fitted nuclear RNA values.
    fit_c : np.ndarray
        Fitted cytoplasmic RNA values.
    Returns:
    --------
    mse_df : pd.DataFrame
        DataFrame containing the log10 MSE ratios for spliced vs nuclear 
        and unspliced vs cytoplasmic RNA, along with the corresponding feature labels.
    """
    # Compute Mean Squared Error (MSE) for spliced, unspliced, nuclear, and cytoplasmic values.
    mse_s = np.mean((fit_s - ms) ** 2, axis=0)
    mse_u = np.mean((fit_u - mu) ** 2, axis=0)
    mse_n = np.mean((fit_n - mn) ** 2, axis=0)
    mse_c = np.mean((fit_c - mc) ** 2, axis=0)

    # Create a DataFrame to store MSE results and log10 ratios.
    mse_df = pd.DataFrame()
    
    # Calculate log10 ratios for spliced/nuclear and unspliced/cytoplasmic MSE.
    sn = np.log10(mse_s / mse_n)
    uc = np.log10(mse_u / mse_c)

    # Add the log10 MSE ratios and feature labels to the DataFrame.
    mse_df["log10 MSE ratio"] = np.concatenate([sn, uc]).ravel()
    mse_df["Feature"] = ["spl-nuc"] * len(sn) + ["uns-cyt"] * len(uc)

    return mse_df


def compute_confidence(adata, vkey="velocity"):
    """
    Compute the velocity confidence for each cell in the given AnnData object and return it as a DataFrame.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix of shape `n_obs` x `n_vars` (cells x genes), 
        typically used in scRNA-seq analysis. This object should include 
        precomputed RNA velocity in the specified layer.
    
    vkey : str, optional (default: "velocity")
        Key in `adata.layers` that contains the RNA velocity values.

    Returns:
    --------
    g_df : pd.DataFrame
        DataFrame containing the velocity confidence for each cell in the dataset, 
        with one column labeled "Velocity confidence".

    Description:
    ------------
    This function computes the RNA velocity graph and the velocity confidence for 
    each cell in the `adata` object using scVelo functions (`scv.tl.velocity_graph` 
    and `scv.tl.velocity_confidence`). It then extracts the computed velocity 
    confidence from `adata.obs` and returns it in a DataFrame.
    """
    # Extract RNA velocity data from the specified layer of the AnnData object.
    velo = adata.layers[vkey]
    
    # Compute the velocity graph based on the RNA velocity data.
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=2)
    
    # Compute the velocity confidence for each cell.
    scv.tl.velocity_confidence(adata, vkey=vkey)
    
    # Create a DataFrame to store the velocity confidence values.
    g_df = pd.DataFrame()
    adata.obs["Velocity confidence"] = adata.obs[f"{vkey}_confidence"].to_numpy().ravel()
    
    return adata

def get_confidence(adata,vkey="velocity"):
    g_df = pd.DataFrame()
    g_df['cell_id']=adata.obs.index
    g_df["Velocity confidence"] = adata.obs[f"{vkey}_confidence"].to_numpy().ravel()
    return g_df


def get_confidences(adata_dict:'AnnData'):
    df = []
    for d in adata_dict.keys():
        conf = get_confidence(adata_dict[d])
        conf["velo_type"] = d
        df.append(conf)
    dfs=pd.concat(df)
    return dfs

def velocity_mean_correlation(adata_dict, mode: str = 'by_gene'):
    """
    Computes the pairwise velocity correlation matrix between different datasets stored in `adata_dict`.

    Parameters:
    -----------
    adata_dict : dict
        A dictionary where keys represent dataset names and values are AnnData objects with velocity data.
    mode : str, optional
        The mode for computing correlations, either 'by_gene' or 'by_cell'. 
        - 'by_gene': Correlations are calculated across genes for each dataset.
        - 'by_cell': Correlations are calculated across cells for each dataset.
        Default is 'by_gene'.

    Returns:
    --------
    corrmat : pandas.DataFrame
        A DataFrame containing the pairwise correlation matrix between the datasets. 
        Each entry represents the average correlation between velocities of either genes or cells (depending on the mode).
    
    Notes:
    ------
    - The function uses masked arrays to handle any invalid (NaN) values in the velocity matrices.
    - `velocity` data is extracted from each AnnData object using `to_df('velocity')`.

    Example:
    --------
    velocity_correlation({'sample1': adata1, 'sample2': adata2}, mode='by_cell')
    """
    
    corrmat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict.keys())
    for d1 in adata_dict.keys():
        for d2 in adata_dict.keys():
            velo1 = adata_dict[d1].to_df('velocity')
            velo2 = adata_dict[d2].to_df('velocity')
            import numpy.ma as ma
            if mode == 'by_cell':
                cross_corrs = [ma.corrcoef(ma.masked_invalid(velo1.iloc[e, :]), ma.masked_invalid(velo2.iloc[e, :]))[0, 1] for e in range(0, velo1.shape[0])]
            if mode == 'by_gene':
                cross_corrs = [ma.corrcoef(ma.masked_invalid(velo1.iloc[:, e]), ma.masked_invalid(velo2.iloc[:, e]))[0, 1] for e in range(0, velo1.shape[1])]
            corrmat.loc[d1, d2] = np.nanmean(cross_corrs)
    
    return corrmat.astype(float)

def velocity_corr_against_reference(adata_dict, reference='adata_s_u_deterministic', mode: str = 'by_gene'):
    """
    Calculate the correlation of velocity values between a reference dataset and multiple datasets.
    
    This function computes the correlation of velocities for each gene or cell between the reference dataset
    and other datasets stored in `adata_dict`. Correlations are computed either on a per-gene or per-cell basis,
    depending on the specified mode.

    Parameters:
    -----------
    adata_dict : dict
        A dictionary where keys are dataset names and values are AnnData objects. The AnnData objects must contain 
        a 'velocity' layer.
    
    reference : str, optional (default: 'adata_s_u_deterministic')
        The key in `adata_dict` corresponding to the reference dataset used for comparison.
    
    mode : str, optional (default: 'by_gene')
        Specifies whether to compute correlations 'by_gene' (correlation of velocities for each gene across cells) 
        or 'by_cell' (correlation of velocities for each cell across genes).
    
    Returns:
    --------
    corrmat : pd.DataFrame
        A DataFrame where rows correspond to the datasets in `adata_dict` and columns correspond to either genes 
        (if mode is 'by_gene') or cells (if mode is 'by_cell'). The values in the DataFrame are the correlation 
        coefficients of the velocity values between the reference dataset and each dataset in `adata_dict`.
    
    Example:
    --------
    >>> corrmat = velocity_corr_against_reference(adata_dict, reference='adata_s_u_deterministic', mode='by_gene')
    """
    d1 = reference

    # Initialize the correlation matrix with appropriate indexing
    if mode == 'by_gene':
        corrmat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict[d1].var.index)
    if mode == 'by_cell':
        corrmat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict[d1].obs.index)

    # Compute correlations for each dataset in adata_dict
    for d2 in adata_dict.keys():
        velo1 = adata_dict[d1].to_df('velocity')  # Extract velocity data from the reference dataset
        velo2 = adata_dict[d2].to_df('velocity')  # Extract velocity data from the current dataset
        
        import numpy.ma as ma  # Use masked arrays to handle NaN values
        
        # Compute correlation by cell (across genes)
        if mode == 'by_cell':
            cross_corrs = [ma.corrcoef(ma.masked_invalid(velo1.iloc[e, :]), ma.masked_invalid(velo2.iloc[e, :]))[0, 1] for e in range(velo1.shape[0])]
        
        # Compute correlation by gene (across cells)
        if mode == 'by_gene':
            cross_corrs = [ma.corrcoef(ma.masked_invalid(velo1.iloc[:, e]), ma.masked_invalid(velo2.iloc[:, e]))[0, 1] for e in range(velo1.shape[1])]
        
        corrmat.loc[d2, :] = cross_corrs  # Store computed correlations in the matrix
    
    return corrmat.astype(float)


def velocity_mean_mse(adata_dict, mode: str = 'by_gene'):
    """
    Computes the pairwise velocity mean squared error (MSE) matrix between different datasets stored in `adata_dict`.

    Parameters:
    -----------
    adata_dict : dict
        A dictionary where keys represent dataset names and values are AnnData objects with velocity data.
    mode : str, optional
        The mode for computing MSE, either 'by_gene' or 'by_cell'. 
        - 'by_gene': MSE is calculated across genes for each dataset.
        - 'by_cell': MSE is calculated across cells for each dataset.
        Default is 'by_gene'.

    Returns:
    --------
    mse_mat : pandas.DataFrame
        A DataFrame containing the pairwise MSE matrix between the datasets. 
        Each entry represents the average MSE between velocities of either genes or cells (depending on the mode).
    
    Notes:
    ------
    - The function handles NaN values using `np.nanmean()` to avoid invalid MSE calculations.
    - `velocity` data is extracted from each AnnData object using `to_df('velocity')`.

    Example:
    --------
    velocity_mean_mse({'sample1': adata1, 'sample2': adata2}, mode='by_cell')
    """
    
    mse_mat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict.keys())
    
    for d1 in adata_dict.keys():
        for d2 in adata_dict.keys():
            velo1 = adata_dict[d1].to_df('velocity')
            velo2 = adata_dict[d2].to_df('velocity')
            import numpy as np
            
            # Compute MSE by cell (across genes)
            if mode == 'by_cell':
                mse_vals = [np.nanmean((velo1.iloc[e, :] - velo2.iloc[e, :]) ** 2) for e in range(velo1.shape[0])]
            
            # Compute MSE by gene (across cells)
            if mode == 'by_gene':
                mse_vals = [np.nanmean((velo1.iloc[:, e] - velo2.iloc[:, e]) ** 2) for e in range(velo1.shape[1])]
            
            # Store the average MSE for the dataset pair
            mse_mat.loc[d1, d2] = np.nanmean(mse_vals)
    
    return mse_mat.astype(float)


def velocity_mse_against_reference(adata_dict, reference='adata_s_u_deterministic', mode: str = 'by_gene'):
    """
    Calculate the mean squared error (MSE) of velocity values between a reference dataset and multiple datasets.
    
    This function computes the MSE for each gene or cell between the reference dataset and other datasets stored 
    in `adata_dict`. MSE is computed either on a per-gene or per-cell basis, depending on the specified mode.

    Parameters:
    -----------
    adata_dict : dict
        A dictionary where keys are dataset names and values are AnnData objects. The AnnData objects must contain 
        a 'velocity' layer.
    
    reference : str, optional (default: 'adata_s_u_deterministic')
        The key in `adata_dict` corresponding to the reference dataset used for comparison.
    
    mode : str, optional (default: 'by_gene')
        Specifies whether to compute MSE 'by_gene' (MSE of velocities for each gene across cells) 
        or 'by_cell' (MSE of velocities for each cell across genes).
    
    Returns:
    --------
    mse_mat : pd.DataFrame
        A DataFrame where rows correspond to the datasets in `adata_dict` and columns correspond to either genes 
        (if mode is 'by_gene') or cells (if mode is 'by_cell'). The values in the DataFrame are the MSE values 
        of the velocity values between the reference dataset and each dataset in `adata_dict`.
    
    Example:
    --------
    >>> mse_mat = velocity_mse_against_reference(adata_dict, reference='adata_s_u_deterministic', mode='by_gene')
    """
    d1 = reference

    # Initialize the MSE matrix with appropriate indexing
    if mode == 'by_gene':
        mse_mat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict[d1].var.index)
    if mode == 'by_cell':
        mse_mat = pd.DataFrame(index=adata_dict.keys(), columns=adata_dict[d1].obs.index)

    # Compute MSE for each dataset in adata_dict
    for d2 in adata_dict.keys():
        velo1 = adata_dict[d1].to_df('velocity')  # Extract velocity data from the reference dataset
        velo2 = adata_dict[d2].to_df('velocity')  # Extract velocity data from the current dataset
        
        import numpy as np  # Import numpy for MSE computation
        
        # Compute MSE by cell (across genes)
        if mode == 'by_cell':
            mse_vals = [np.nanmean((velo1.iloc[e, :] - velo2.iloc[e, :]) ** 2) for e in range(velo1.shape[0])]
        
        # Compute MSE by gene (across cells)
        if mode == 'by_gene':
            mse_vals = [np.nanmean((velo1.iloc[:, e] - velo2.iloc[:, e]) ** 2) for e in range(velo1.shape[1])]
        
        mse_mat.loc[d2, :] = mse_vals  # Store computed MSE in the matrix
    
    return mse_mat.astype(float)



def get_classification_scores(velo_sign_true, velo_sign_pred, score_fun, **kwargs):
    n_vars = velo_sign_true.shape[1]
    em_score = [
        score_fun(velo_sign_true[:, var_id], velo_sign_pred["EM"][:, var_id], **kwargs)
        for var_id in range(n_vars)
    ]
    return em_score

def sign_accuracy(adata):
    aggr_counts = []
    for pos in tqdm(np.sort(adata.obs["true_t"].unique())):
        mask = (adata.obs["true_t"] == pos).values

        aggr_counts.append(np.median(adata.layers["Ms"][mask, :], axis=0))

    aggr_counts = np.vstack(aggr_counts)

    reorder_mask = np.arange(1, adata.obs["true_t"].nunique()).tolist() + [0]

    cell_cycle_pos = adata.obs["true_t"].values
    cc_pos_diff = np.sort(np.unique(cell_cycle_pos))

    cc_pos_diff = (cc_pos_diff[reorder_mask] - cc_pos_diff) % (2 * np.pi)

    empirical_velo = (aggr_counts[reorder_mask, :] - aggr_counts) / cc_pos_diff.reshape(-1, 1)

    empirical_velo_sign = np.sign(empirical_velo)
    aggr_counts = []
    for pos in tqdm(np.sort(adata.obs["true_t"].unique())):
        mask = (adata.obs["true_t"] == pos).values

        aggr_counts.append(np.median(adata.layers["Ms"][mask, :], axis=0))

    aggr_counts = np.vstack(aggr_counts)
    np.random.seed(0)
    random_velo_sign = np.random.choice([-1, 0, 1], size=(len(adata.obs["true_t"].unique()), adata.n_vars))
    baseline_performance = [
        accuracy_score(empirical_velo_sign[:, var_id], random_velo_sign[:, var_id])
        for var_id in range(adata.n_vars)
    ]
    aggr_velo = {"EM": []}

    for pos in tqdm(np.sort(adata.obs["true_t"].unique())):
        mask = (adata.obs["true_t"] == pos).values
        aggr_velo["EM"].append(np.median(adata.layers["velocity"][mask, :], axis=0))
    aggr_velo["EM"] = np.vstack(aggr_velo["EM"])
    aggr_velo["EM"]= np.nan_to_num(aggr_velo["EM"],nan=0.0)
    aggr_velo_sign = {}
    aggr_velo_sign["EM"] = np.sign(aggr_velo["EM"])
    
    em_score = get_classification_scores(
        velo_sign_true=empirical_velo_sign,
        velo_sign_pred=aggr_velo_sign,score_fun=accuracy_score)
    return em_score

def get_sign_accuracy(adata_dict):
    """
    Computes the sign accuracy for velocity models in a dictionary of AnnData objects.

    Parameters:
    -----------
    adata_dict : dict
        A dictionary where keys are dataset names and values are AnnData objects containing velocity data.
    
    Returns:
    --------
    allsigns : pd.DataFrame
        A concatenated DataFrame with sign accuracy scores for each dataset, including the dataset name as 'Model'.
    """
    allsig = []
    for k in adata_dict.keys():
        print(k)
        adata = adata_dict[k]
        em_score = sign_accuracy(adata)  # Assumes sign_accuracy function is defined elsewhere
        df = pd.DataFrame({"Accuracy": em_score, "Model": [k] * adata.n_vars})
        allsig.append(df)
    
    allsigns = pd.concat(allsig)
    return allsigns

