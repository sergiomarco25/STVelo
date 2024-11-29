import scvelo as scv
from scvi.external import VELOVI
import torch
import numpy as np
from pipelines.preprocessing import preprocess_data_velovi

from scipy.sparse import csr_matrix, issparse

from scvelo.pp.utils import not_yet_normalized, normalize_per_cell
from scvelo.pp.neighbors  import get_connectivities

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
        self.velovi_model_params = config.get('velovi_model_params',None)
        self.velovi_train_params = config.get('velovi_train_params',None)
        self.velovi_getting_parameters = config.get('velovi_getting_parameters',None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

    
    def compute_velocities(self):
        result_adatas = {}  
        if 'velovi' in self.velocity_types:
            vae_dict = {}
        for name, adata in self.adatas.items():
            if 'n_c' in name:
                min_r2 = 0.01
            else:
                min_r2 = 0.01
            # Extract the part after the first '_' in the name
            idx = name.split('_', 1)[-1] if '_' in name else name
            for velocity_type in self.velocity_types:
                
                adata_copy = adata.copy()
                if velocity_type in ['deterministic', 'stochastic', 'dynamical']:
                    # For scVelo velocity modes
                    if velocity_type == 'dynamical':
                        n_jobs = self.config.get('n_jobs',1)
                        print(n_jobs)
                        scv.tl.recover_dynamics(adata_copy)
                    print(f'{velocity_type} velocity is being calculated.')
                    scv.tl.velocity(adata_copy, mode=velocity_type,min_r2=min_r2)
                    scv.tl.velocity_graph(adata_copy)
                    key = f'adata_{idx}_{velocity_type}'
                    result_adatas[key] = adata_copy

                elif velocity_type == 'velovi':
                    print(name,min_r2)
                    adata_copy = preprocess_data_velovi(adata_copy,min_r2=min_r2)
                    print('min_max_scaler is working!')
                    
                    VELOVI.setup_anndata(adata_copy, spliced_layer="Ms", unspliced_layer="Mu")

                    n_hidden = self.velovi_model_params.get('n_hidden',256)
                    n_latent = self.velovi_model_params.get('n_latent',10)
                    n_layers = self.velovi_model_params.get('n_layers',1)


                    vae = VELOVI(adata_copy, n_hidden=n_hidden, n_latent=n_latent,n_layers=n_layers)
                    vae.to_device(self.device)
                    
                    epochs = self.velovi_train_params.get('epochs',200)
                    lr = self.velovi_train_params.get('lr',0.01)
                    weight_decay = self.velovi_train_params.get('weight_decay',0.01)
                    early_stop = self.velovi_train_params.get('early_stop',True)
                    batch_size = self.velovi_train_params.get('batch_size',256)

                    vae.train(max_epochs=epochs, lr= lr, weight_decay=weight_decay, early_stopping= early_stop, batch_size=batch_size)
                    scv.tl.velocity_graph(adata_copy,n_jobs=1)

                    self.add_velovi_outputs_to_adata(adata_copy,vae)

                    key = f'adata_{idx}_velovi'
                    vae_dict[key] = vae
                    result_adatas[key] = adata_copy

                else:
                    print(f"Unknown velocity type: {velocity_type}")
        if 'velovi' in self.velocity_types:
            return vae_dict, result_adatas
        else:
            return result_adatas

    def add_velovi_outputs_to_adata(self, adata, vae):
        t_max = self.velovi_getting_parameters.get('t_max',30)
        print(f't_max:{t_max}')
        n_samples = self.velovi_getting_parameters.get('n_samples', 25)
        latent_time = vae.get_latent_time(n_samples=n_samples )
        velocities = vae.get_velocity(n_samples=n_samples, velo_statistic="mean")

        t = latent_time
        scaling = t_max / t.max(0)
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



    def moments_3(adata_dict):
        for k, adata in adata_dict.items():
            if 's_u' in k:
                layers = [layer for layer in {"spliced_cyt", "spliced_nuc"} if layer in adata.layers]
                if any([not_yet_normalized(adata.layers[layer]) for layer in layers]):
                    normalize_per_cell(adata)
                connectivities = get_connectivities(adata, mode= 'connectivities', recurse_neighbors=False)

                adata.layers['Msn'] = (
                        csr_matrix.dot(connectivities, csr_matrix(adata.layers["spliced_nuc"]))
                        .astype(np.float32)
                        .toarray()
                        )
                
                adata.layers['Msc'] = (
                        csr_matrix.dot(connectivities, csr_matrix(adata.layers["spliced_cyt"]))
                        .astype(np.float32)
                        .toarray()
                        )
                
            if 'n_c' in k:
                    layers = [layer for layer in {"spliced_cyt", "spliced_nuc"} if layer in adata.layers]
                    if any([not_yet_normalized(adata.layers[layer]) for layer in layers]):
                        normalize_per_cell(adata)
                    connectivities = get_connectivities(adata, mode= 'connectivities', recurse_neighbors=False)

                    adata.layers['Msn'] = (
                        csr_matrix.dot(connectivities, csr_matrix(adata.layers["spliced_nuc"]))
                        .astype(np.float32)
                        .toarray()
                        )
                
                    adata.layers['Msc'] = (
                        csr_matrix.dot(connectivities, csr_matrix(adata.layers["spliced_cyt"]))
                        .astype(np.float32)
                        .toarray()
                        )


    def calculate_true_velocity_on_moments(self,adata_dict):
        for k, adata in adata_dict.items():
            if 's_u' in k:
                beta = adata.var['true_beta']
                beta = np.array(beta)

                gamma = adata.var['true_gamma']
                gamma = np.array(gamma)

                Mu = adata.layers['Mu']
                Msc = adata.layers['Msc']

                velocity = (beta * Mu) - (gamma * Msc)

                adata.layers['true_velocity_moment'] = velocity 

            if 'n_c' in k:
                nu = adata.var['true_nu']
                nu = np.array(nu)

                gamma = adata.var['true_gamma']
                gamma = np.array(gamma)

                Msn = adata.layers['Msn']
                Msc = adata.layers['Msc']

                velocity = (nu * Msn) - (gamma * Msc)  

                adata.layers['true_velocity_moment'] = velocity 


