
class Plotter:
    def __init__(self, adata_dict, config, saving_path='./plots', saving_format='svg'):
        import os
        
        self.adata_dict = adata_dict
        self.config = config
        self.saving_path = saving_path
        self.saving_format = saving_format
        
        # Ensure the saving path exists
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
    
    def generate_plots(self, save=True):
        import os
        import scvelo as scv
        import matplotlib.pyplot as plt
        import pandas as pd

        for d in self.adata_dict.keys():
            adata = self.adata_dict[d]
            saving_adata_path = os.path.join(self.saving_path, d)
            if not os.path.exists(saving_adata_path):
                os.makedirs(saving_adata_path)

            # Get the list of colorsets from the config, defaulting to ['leiden']
            colorsets = self.config.get('colorsets', ['leiden'])
            for colorset in colorsets:
                # Velocity Embedding Stream Plot
                if self.config.get('velocity_embedding_stream', False):
                    try:
                        scv.pl.velocity_embedding_stream(adata, basis='umap', color=colorset, show=False)
                        if save:
                            plt.savefig(os.path.join(saving_adata_path, f'UMAP_{colorset}_velocity_stream.{self.saving_format}'))
                        plt.close()
                    except Exception as e:
                        print(f"An error occurred while generating velocity_embedding_stream for {colorset}: {e}")

                # Velocity Embedding Grid Plot
                if self.config.get('velocity_embedding_grid', False):
                    try:
                        scv.pl.velocity_embedding_grid(adata, basis='umap', color=colorset, show=False)
                        if save:
                            plt.savefig(os.path.join(saving_adata_path, f'UMAP_{colorset}_velocity_grid.{self.saving_format}'))
                        plt.close()
                    except Exception as e:
                        print(f"An error occurred while generating velocity_embedding_grid for {colorset}: {e}")

                # Velocity Embedding Plot
                if self.config.get('velocity_embedding', False):
                    try:
                        scv.pl.velocity_embedding(
                            adata,
                            basis='umap',
                            arrow_length=3,
                            color=colorset,
                            arrow_size=2,
                            dpi=120,
                            show=False
                        )
                        if save:
                            plt.savefig(os.path.join(saving_adata_path, f'UMAP_{colorset}_velocity_embedding.{self.saving_format}'))
                        plt.close()
                    except Exception as e:
                        print(f"An error occurred while generating velocity_embedding for {colorset}: {e}")

            # Rank Velocity Genes and Plot
            if self.config.get('rank_velocity_genes', False):
                try:
                    scv.tl.rank_velocity_genes(adata, groupby='leiden', min_corr=.3)
                    df = pd.DataFrame(adata.uns['rank_velocity_genes']['names'])
                    for col in df.columns:
                        # Scatter Plot
                        scv.pl.scatter(adata, df[col][:5], ylabel=col, color='leiden', show=False)
                        if save:
                            plt.savefig(os.path.join(saving_adata_path, f'scatterplot_velo_{col}.{self.saving_format}'))
                        plt.close()

                        # Velocity Plot
                        scv.pl.velocity(adata, df[col][:5], ncols=2, add_outline=True, show=False)
                        if save:
                            plt.savefig(os.path.join(saving_adata_path, f'velocity_plots_{col}.{self.saving_format}'))
                        plt.close()
                except Exception as e:
                    print(f"An error occurred while generating rank_velocity_genes plots: {e}")

            # Velocity Confidence and Length Plots
            if self.config.get('velocity_confidence', False) or self.config.get('velocity_length', False):
                try:
                    scv.tl.velocity_confidence(adata)
                except Exception as e:
                    print(f"An error occurred while computing velocity_confidence: {e}")

            if self.config.get('velocity_confidence', False):
                try:
                    scv.pl.scatter(adata, c='velocity_confidence', cmap='coolwarm', perc=[5, 95], show=False)
                    if save:
                        plt.savefig(os.path.join(saving_adata_path, f'umap_confidence.{self.saving_format}'))
                    plt.close()
                except Exception as e:
                    print(f"An error occurred while generating velocity_confidence plot: {e}")

            if self.config.get('velocity_length', False):
                try:
                    scv.pl.scatter(adata, c='velocity_length', cmap='coolwarm', perc=[5, 95], show=False)
                    if save:
                        plt.savefig(os.path.join(saving_adata_path, f'umap_velocity_length.{self.saving_format}'))
                    plt.close()
                except Exception as e:
                    print(f"An error occurred while generating velocity_length plot: {e}")
