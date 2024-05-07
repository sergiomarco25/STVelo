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

def subcellular_featurization(adatafilt,knn=3):
    '''Function to featurize the cells but using the subcellular location of transcripts instead'''
    from tqdm import tqdm
    from libpysal import weights, examples
    from contextily import add_basemap
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree

    tr2=adatafilt.uns['spots']
    tr2['feature_name']=tr2['feature_name'].str.replace('_','')
    allcells=tr2['cell_id'].unique()
    # modify you inputdf data frame with all info specified below
    input_df=tr2
    inputdf=tr2
    # the input is a dataframe with info individual of every read in rows should contain:
    # x,y and z position (make a dummy z if you don't have it) stored in 'x_location','y_location' and 'z_location'
    # a "cell_id" column, whichi indicate the cell each read has been assigned to
    # a "feature_name" column, containing the "gene" of each transcript

    allcells=len(input_df['cell_id'].unique())
    step=1000
    alladata=[]
    for i in tqdm(range(0,int(np.ceil(allcells/step)))):
        print(i)
        selectedcells=inputdf['cell_id'].unique()[i*step:(i*step)+step]
        cell=inputdf.loc[inputdf['cell_id'].isin(selectedcells),:]
        cell=cell.reset_index()
        dictio=dict(zip(cell.index,cell['feature_name']+'_'+cell['cell_id'].astype(str)))
        # Coordinates of the points
        points =np.array(cell.loc[:,['x_location','y_location']]) #,'z_location'
        # Create the KDTree
        tree = KDTree(points)

        # Find the indices of the points within the radius of each point
        indices = tree.query(points, k=knn)
        commong=np.vectorize(dictio.get)(indices[1])
        print('network created')
        alli=[]
        allids=cell['feature_name']+'_'+cell['cell_id'].astype(str)
        for c in range(1,commong.shape[1]):
            alli.append(pd.DataFrame({'feature_name':allids,'neigh':commong[:,c]}))
        alli2=pd.concat(alli)
        del allids
        del alli
        del commong
#        print(alli2)
#        print('splitting columns')
        alli2[['gene1','cell1']]=alli2['feature_name'].str.split('_',expand=True)
        alli2[['gene2','cell2']]=alli2['neigh'].str.split('_',expand=True)
        alli2.drop('feature_name',inplace=True,axis=1)
        alli2.drop('neigh',inplace=True,axis=1)
        alli2=alli2[alli2['cell1']==alli2['cell2']]
        alli2.drop('cell2',inplace=True,axis=1)
        alli2['interaction']=alli2['gene1']+'_'+alli2['gene2']
        alli2.drop('gene1',inplace=True,axis=1)
        alli2.drop('gene2',inplace=True,axis=1)
        print('cross tabulation')
        data = alli2.groupby(['cell1', 'interaction']).size().reset_index(name='count')
        els=np.unique(data['interaction'])
        unicode=dict(zip(els,range(0,len(els))))
        unicoderev=dict(zip(range(0,len(els)),els))
        data['interaction']=data.interaction.map(unicode)
        ela=np.unique(data['cell1'])
        unicell=dict(zip(ela,range(0,len(ela))))
        unicellrev=dict(zip(range(0,len(ela)),ela))
        data['cell1']=data.cell1.map(unicell)
        from scipy.sparse import coo_matrix
        sparse_matrix = coo_matrix((data['count'], (data['cell1'].astype(int),data['interaction'].astype(int))))
        print('creating_adata')
        adataneigh=sc.AnnData(sparse_matrix)
        adataneigh.obs.index=adataneigh.obs.index.astype(int).map(unicellrev)
        adataneigh.var.index=adataneigh.var.index.astype(int).map(unicoderev)
        alladata.append(adataneigh)
    adataneigh=sc.concat(alladata,join='outer')
return adataneigh