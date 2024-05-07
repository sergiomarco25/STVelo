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



def polarity_visualizer(adatafilt:AnnData,cell_id_sel='',num=3,clust='leiden',gap=30):
    xcen=adatafilt.obs.loc[cell_id_sel,'x_centroid']
    ycen=adatafilt.obs.loc[cell_id_sel,'y_centroid']
    cell_reads=adatafilt.uns['spots'][adatafilt.uns['spots']['cell_id']==cell_id_sel]
    nuc_reads=cell_reads[cell_reads['overlaps_nucleus']==1]
    poly=adatafilt.obsm['polarity'].loc[adatafilt.obsm['polarity'].index==cell_id_sel].transpose()
    selg=poly.sort_values(by=cell_id_sel,ascending=False).index[0:num]
    plt.figure(figsize=(4,4))
    from scipy.spatial import ConvexHull
    points=np.array([cell_reads['x_location'],cell_reads['y_location']]).transpose()
    rds=ConvexHull(points)
    plt.plot(points[rds.vertices,0], points[rds.vertices,1], 'r--', lw=2)
    plt.scatter(cell_reads['x_location'],cell_reads['y_location'],c='grey',s=2)
    plt.scatter(nuc_reads['x_location'],nuc_reads['y_location'],c='black',s=2)
    flt=cell_reads[cell_reads['feature_name'].isin(selg)]
    flt=flt.sort_values(by='feature_name')
    sns.scatterplot(flt['x_location'],flt['y_location'],hue=flt['feature_name'])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.scatter(xcen,ycen,marker='+',s=10,color='red')
    plt.axis('off')
    plt.title(str(cell_id_sel)+'_cl'+str(adatafilt.obs.loc[cell_id_sel,clust]))
    rf=adatafilt.uns['spots'][adatafilt.uns['spots']['x_location']>xcen-gap]
    rf=rf[rf['x_location']<xcen+gap]
    rf=rf[rf['y_location']<ycen+gap]
    rf=rf[rf['y_location']>ycen-gap]
    rf=rf[rf['cell_id']!=cell_id_sel]
    plt.scatter(rf['x_location'],rf['y_location'],c='#ffac2a',s=0.2)
    flt=rf[rf['feature_name'].isin(selg)]
    flt=flt.sort_values(by='feature_name')
    sns.scatterplot(flt['x_location'],flt['y_location'],hue=flt['feature_name'],style=3,s=4)
    plt.legend(bbox_to_anchor=(1.5, 1), loc='upper left', borderaxespad=0)
    plt.show()
