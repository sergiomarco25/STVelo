import scvelo as scv
import scanpy as sc
import pandas as pd
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import json

class VelocityAnalysis:
    def __init__(self, config):
        print(config)
        self.data_path = config["data_path"]
        self.mode = config["mode"]
        self.preprocess_params = config["preprocess_params"]
        self.save_anndata = config["save_anndata"]
        self.adata = None
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        self.adata = scv.read(self.data_path)
        self.adata.obsm["spatial"] = self.adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    def preprocess_data(self):
        print("preprocessing data")
        sc.pp.filter_cells(self.adata, min_counts=self.preprocess_params["min_counts"])
        sc.pp.filter_genes(self.adata, min_cells=self.preprocess_params["min_cells"])
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata, n_neighbors=self.preprocess_params["n_neighbors"], n_pcs=self.preprocess_params["n_pcs"])
        sc.tl.umap(self.adata, min_dist=self.preprocess_params["min_dist"])
        sc.tl.leiden(self.adata, resolution=self.preprocess_params["resolution"])
        scv.pp.moments(self.adata, n_pcs=None, n_neighbors=None)

    def calculate_velocity(self):
        if self.mode == 'dynamical':
            scv.tl.recover_dynamics(self.adata)
        scv.tl.velocity(self.adata, mode=self.mode)
        scv.tl.velocity_graph(self.adata)

    def plot_velocity_embedding(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scv.pl.velocity_embedding_stream(self.adata, basis='umap', color='leiden', dpi=110, show=False, save=f'{self.output_dir}/velocity_embedding_stream.jpg')

    def plot_spatial(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pl.spatial(self.adata, color="leiden", spot_size=12, show = False ,save=f'{self.output_dir}/spatial.jpg')
            #plt.savefig(f'{self.output_dir}/spatial_plot.jpg', dpi=300, bbox_inches='tight')


    def plot_histograms(self):
        df = self.adata.var
        df = df[(df['fit_likelihood'] > .1) & (df['velocity_genes'] == True)]
        kwargs = dict(fontsize=16)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].hist(df['fit_alpha'], bins=50)
        axs[0].set_xlabel('transcription rate', **kwargs)
        axs[0].set_xscale('log')

        axs[1].hist(df['fit_beta'] * df['fit_scaling'], bins=50)
        axs[1].set_xlabel('splicing rate', **kwargs)
        axs[1].set_xscale('log')
        axs[1].set_xticks([.1, .4, 1])

        axs[2].hist(df['fit_gamma'], bins=50)
        axs[2].set_xlabel('degradation rate', **kwargs)
        axs[2].set_xscale('log')
        axs[2].set_xticks([.1, .4, 1])

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/histograms.jpg')
    

    def save_results(self):
        df_fit_gut = scv.get_df(self.adata, 'fit*', dropna=True)
        df_fit_gut.to_csv(os.path.join(self.output_dir, 'df_fit_gut.csv'), index=False)


    def save_data(self):
        if self.save_anndata:
            self.adata.write(os.path.join(self.output_dir, 'adata_processed.h5ad'))
            data_path = os.path.join(self.output_dir, 'adata_processed.h5ad')
            print(f'anndata object saved to {data_path}')
        else:
            print("anndata not saved, set save_anndata = true")

    def run_analysis(self):
        self.load_data()
        self.preprocess_data()
        self.calculate_velocity()
        #self.plot_velocity_embedding()
        #self.plot_spatial()
        if self.mode == 'dynamical':
            self.plot_histograms()
            #self.save_results()

        self.save_data()

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    analysis = VelocityAnalysis(config)
    analysis.run_analysis()

if __name__ == "__main__":
    config_path = input("Config file path: ")
    main(config_path)

