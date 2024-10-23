import scvelo as scv
from scvi.external import VELOVI
import torch
import numpy as np
from pipelines.preprocessing import preprocess_data_velovi

class Velocities:
    def __init__(self, adatas, config):
        """
        Parameters:
        - adatas (dict): A dictionary where keys are names (e.g., 'adata_s_u') and values are AnnData objects.
        - config (dict): Configuration dictionary specifying which velocity models to apply.
        """
        self.adatas = adatas  # Expecting a dictionary {name: adata}
        self.config = config
        self.velocity_types = config.get('velocity_types', [])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

    
    def compute_velocities(self):
        result_adatas = {}  
        for name, adata in self.adatas.items():

            # Extract the part after the first '_' in the name
            idx = name.split('_', 1)[-1] if '_' in name else name
            for velocity_type in self.velocity_types:
                
                adata_copy = adata.copy()
                if velocity_type in ['deterministic', 'stochastic', 'dynamical']:
                    # For scVelo velocity modes
                    if velocity_type == 'dynamical':
                        scv.tl.recover_dynamics(adata_copy,n_jobs=8)
                    print(f'{velocity_type} velocity is being calculated.')
                    scv.tl.velocity(adata_copy, mode=velocity_type)
                    scv.tl.velocity_graph(adata_copy,n_jobs=8)
                    key = f'adata_{idx}_{velocity_type}'
                    result_adatas[key] = adata_copy

                elif velocity_type == 'velovi':
                    
                    adata_copy = preprocess_data_velovi(adata_copy)
                    print('min_max_scaler is working!')
                    VELOVI.setup_anndata(adata_copy, spliced_layer="Ms", unspliced_layer="Mu")
                    vae = VELOVI(adata_copy)
                    vae.to_device(self.device)
                    vae.train(max_epochs=100)

                    self.add_velovi_outputs_to_adata(adata_copy,vae)

                    key = f'adata_{idx}_velovi'
                    result_adatas[key] = adata_copy

                else:
                    print(f"Unknown velocity type: {velocity_type}")
        return result_adatas

    def add_velovi_outputs_to_adata(self, adata, vae):
        latent_time = vae.get_latent_time(n_samples=25)
        velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

        t = latent_time
        scaling = 20 / t.max(0)
        scaling = np.array(scaling)
        

        adata.layers["velocity"] = velocities / scaling
        adata.layers["latent_time_velovi"] = latent_time

        adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
        adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
        adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
        adata.var["fit_t_"] = (
            torch.nn.functional.softplus(vae.module.switch_time_unconstr)
            .detach()
            .cpu()
            .numpy()
        ) * scaling
        adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
        adata.var['fit_scaling'] = 1.0