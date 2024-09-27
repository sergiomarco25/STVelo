import os
from pathlib import Path
import sys
import numpy as np
import scvelo as scv
import pandas as pd
from tqdm import tqdm


def apply_2d_segmentation_artifacts(adata, cell_radius=3, section_thickness=6, 
                                       n_points=10000, proportion_of_cells_cut=0.5, mean_cytoplasmic=0.3):
    """
    Simulate potential 2D segmentation artifacts by modeling the interaction between a sphere (nucleus)
    and a cylinder (cell) under random cuts or slices along the z-axis, and estimate misassigned points.

    Parameters:
    ----------
    adata: Anndata
        AnnData with simulated cells
    cell_radius : float, optional
        Radius of the spherical nucleus (default is 3).
    section_thickness : float, optional
        Thickness of the section profiled (default is 6).
    n_points : int, optional
        Number of random points to simulate per run (default is 10,000).
    proportion_of_cells_cut : float, optional
        Proportion of the cell that's included in the section cut (default is 0.5).
    mean_cytoplasmic : float, optional
        Mean proportion of clear cytoplasm (default is 0.3).

    Returns:
    --------
    adata: AnnData
        Updated AnnData object with 2D segmentation artifacts applied.
    """
    
    # Check for valid proportion_of_cells_cut to avoid negative cutting plane
    if proportion_of_cells_cut <= 0 or proportion_of_cells_cut > 1:
        raise ValueError("proportion_of_cells_cut must be between 0 and 1")

    # Copy layers to avoid direct modification of the original data
    unsp = adata.layers['unspliced'].copy()
    sp = adata.layers['spliced'].copy()
    
    # Lists to store results for each simulation
    all_fracts = []
    all_nucleifract = []
    all_cytoprop = []
    all_nuclei_included_prop = []

    # Perform Monte Carlo simulations
    for r in tqdm(range(adata.shape[0]), desc="Simulating missegmentation artifacts"):

        # Randomly determine the proportion of clear cytoplasm using normal distribution
        clear_cytoplasm_proportion = np.random.normal(mean_cytoplasmic, 0.05, 1)
        

        
        # Generate random points inside the cylinder
        z_points = np.random.uniform(-section_thickness / 2, section_thickness / 2, n_points)
        x_points = np.random.uniform(-cell_radius, cell_radius, n_points)
        y_points = np.random.uniform(-cell_radius, cell_radius, n_points)

        # Calculate the cutting plane limits
        cutmax = ((section_thickness / 2) / proportion_of_cells_cut) - section_thickness
        
        # Debug: Check if cutmax is valid
        if cutmax < section_thickness / 2:
            raise ValueError(f"cutmax ({cutmax}) must be greater than section_thickness / 2")
        
        # Randomly determine the cutting plane position
        cutting = np.random.uniform(section_thickness / 2, cutmax, 1)


        # Calculate the proportion of nuclei included in the cut
        included_prop = np.sum(z_points < cutting[0]) / len(z_points)
        
        # Filter points below the cutting plane
        y_points = y_points[z_points < cutting[0]]
        x_points = x_points[z_points < cutting[0]]
        z_points = z_points[z_points < cutting[0]]

        # Equation of the cylinder (x^2 + y^2 <= r_cylinder^2)
        inside_cylinder = x_points**2 + y_points**2 <= cell_radius**2

        # Equation of the sphere (x^2 + y^2 + z^2 <= r_sphere^2)
        inside_sphere = x_points**2 + y_points**2 + z_points**2 <= cell_radius**2

        # Intersection between the cylinder and the sphere
        intersection = inside_cylinder & inside_sphere
        other_cytoplasm_points = len(z_points) * clear_cytoplasm_proportion
        
        # Calculate the fraction of points that are not in the intersection (exclusive to one region)
        fraction_exclusive = ((n_points - np.sum(intersection)) / n_points)

        # Calculate the fraction of misassigned points relative to cytoplasm
        fraction_missassigned = ((n_points - np.sum(intersection)) / ((n_points - np.sum(intersection)) + other_cytoplasm_points))
        

        # Update the unspliced and spliced layers
        frac = sp[r, :] * fraction_missassigned[0]
        unsp[r, :] = unsp[r, :] + frac
        sp[r, :] = sp[r, :] - frac

        # Append results for this simulation
        all_fracts.append(fraction_missassigned[0])
        all_nucleifract.append(fraction_exclusive)
        all_cytoprop.append(clear_cytoplasm_proportion[0])
        all_nuclei_included_prop.append(included_prop)
     
    # Update adata with modified layers
    adata.layers['unspliced'] = unsp
    adata.layers['spliced'] = sp
    adata.X = unsp + sp  # Update the total expression matrix

    # Compile results into a pandas DataFrame
    allsim = pd.DataFrame({
        'cyto_missassigned_fraction': all_fracts,
        'nuclei_missassigned_fraction': all_nucleifract,
        'clear_cytoplasm_proportion': all_cytoprop,
        'proportion_of_nuclei_included': all_nuclei_included_prop
    })
    
    allsim.index = adata.obs.index
    adata.obs = pd.concat([adata.obs, allsim], axis=1)
    
    return adata


def add_2d_missegmentation_artifacts(adata_dict: 'AnnData', cell_radius=10, section_thickness=10, 
                                     mean_cytoplasmic=0.3, proportion_of_cells_cut=0.3):
    """
    Apply 2D missegmentation artifacts to multiple AnnData objects.
    
    Parameters:
    ----------
    adata_dict : dict of AnnData
        A dictionary where keys are names of AnnData objects and values are the corresponding AnnData objects.
    cell_radius : float, optional
        Radius of the spherical nucleus (default is 10).
    section_thickness : float, optional
        Thickness of the section being profiled (default is 10).
    mean_cytoplasmic : float, optional
        Mean proportion of clear cytoplasm (default is 0.3).
    proportion_of_cells_cut : float, optional
        Proportion of the cell that's included in the section cut (default is 0.3).

    Returns:
    --------
    adata_dict : dict of AnnData
        The updated dictionary with modified AnnData objects containing 2D missegmentation artifacts.
    """
    
    ndic = {}
    for key in adata_dict.keys():
        ndic[key + '_2dseg'] = apply_2d_segmentation_artifacts(
            adata_dict[key].copy(), 
            cell_radius=cell_radius, 
            section_thickness=section_thickness, 
            n_points=10000, 
            proportion_of_cells_cut=proportion_of_cells_cut, 
            mean_cytoplasmic=mean_cytoplasmic
        )
    
    adata_dict.update(ndic)
    return adata_dict
