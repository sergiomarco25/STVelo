import numpy as np
import warnings
from anndata import AnnData
import os


class Simulation3ODE:
    def __init__(
        self,
        config,
        random_seed=0,
        
    ):
        """
        Initializes the Simulation3ODE class with the given parameters.
        """
        self.config = config
        params = self.config.get('parameters',{})
        
        options = self.config.get('options',{})

        coeff_options = self.config.get('coeff_generate_options',{})


        self.n_obs = params.get('n_obs', 300)
        self.n_vars = params.get('n_vars', 100)
        self.alpha_ = params.get('alpha_', 0)


        if options.get('generate_parameters'):
            corr = coeff_options.get('corr')
            sd = coeff_options.get('sd')
            means = coeff_options.get('mean')

            self.alpha, self.beta, self.nu, self.gamma = self.sample_parameters(correlations=corr, sigmas=sd, means=means, n_samples= self.n_vars)
        
        else:
            self.alpha = params.get('alpha', 5)
            self.beta = params.get('beta', 0.6)
            self.nu = params.get('nu', 0.3)
            self.gamma = params.get('gamma', 0.25)
        
        self.t_max = params.get('t_max', None)
        self.noise_model = params.get('noise_model', "normal")
        self.noise_level = params.get('noise_level', 1)
        
        self.random_seed = random_seed

        self.save = options.get("save",False)
        self.saving_path = options.get("saving_path",None)
        self.generate_switch_times = options.get("generate_switch_times", False)

        np.random.seed(self.random_seed)

        # Ensure saving_path exists if save is True
        if self.save and self.saving_path is None:
            raise ValueError("Please provide a 'saving_path' when 'save' is True.")

    def unspliced(self, tau, u0, alpha, beta):
        expu = np.exp(-beta * tau)
        return (u0 * expu) + ((alpha / beta) * (1 - expu))

    def spliced_nucleus(self, tau, sn0, alpha, nu, beta, u0):
        term1 = sn0 * np.exp(-nu * tau)
        term2 = (alpha / nu) * (1 - np.exp(-nu * tau))
        term3 = (
            (alpha - beta * u0) / (nu - beta)
        ) * (np.exp(-nu * tau) - np.exp(-beta * tau))
        return term1 + term2 + term3

    def spliced_cyto(self, tau, alpha, beta, gamma, nu, u0, sn0, sc0):
        exp_beta = np.exp(-beta * tau)
        exp_gamma = np.exp(-gamma * tau)
        exp_nu = np.exp(-nu * tau)

        term1 = (alpha / beta) * (1 - exp_beta) + (u0 * exp_beta)
        term1 *= (nu * beta) / ((nu - beta) * (gamma - beta))

        term2 = (alpha / nu) * (1 - exp_nu) + (u0 * exp_nu)
        term2 *= (nu * beta) / ((gamma - nu) * (nu - beta))

        term3 = (alpha / gamma) * (1 - exp_gamma) + (u0 * exp_gamma)
        term3 *= (nu * beta) / ((gamma - nu) * (gamma - beta))

        term4 = (nu / (gamma - nu)) * (exp_nu - exp_gamma) * sn0
        term5 = exp_gamma * sc0

        result = term1 - term2 + term3 + term4 + term5

        return result

    def vectorize(
        self,
        t,
        t_,
        alpha,
        beta,
        nu,
        gamma=None,
        alpha_=0,
        u0=0,
        sn0=0,
        sc0=0,
        sorted=False,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = np.array(t < t_, dtype=int)
        tau = t * o + (t - t_) * (1 - o)

        u0_ = self.unspliced(t_, u0, alpha, beta)
        sn0_ = self.spliced_nucleus(t_, sn0, alpha, nu, beta, u0)
        sc0_ = self.spliced_cyto(t_, alpha, beta, gamma, nu, u0, sn0, sc0)

        # Vectorize u0, sn0, sc0, and alpha
        u0 = u0 * o + u0_ * (1 - o)
        sn0 = sn0 * o + sn0_ * (1 - o)
        sc0 = sc0 * o + sc0_ * (1 - o)
        alpha = alpha * o + alpha_ * (1 - o)

        if sorted:
            idx = np.argsort(t)
            tau, alpha, u0, sn0, sc0 = tau[idx], alpha[idx], u0[idx], sn0[idx], sc0[idx]
        return tau, alpha, u0, sn0, sc0

    def simulate_dynamics(
        self, tau, alpha, beta, nu, gamma, u0, sn0, sc0, noise_model, noise_level
    ):
        ut = self.unspliced(tau, u0, alpha, beta)
        snt = self.spliced_nucleus(tau, sn0, alpha, nu, beta, u0)
        sct = self.spliced_cyto(tau, alpha, beta, gamma, nu, u0, sn0, sc0)

        if noise_model == "normal":  # Add noise
            ut += np.random.normal(
                scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
            )
            snt += np.random.normal(
                scale=noise_level * np.percentile(snt, 99) / 10, size=len(snt)
            )

            percentile_99 = np.percentile(sct, 99)
            if percentile_99 < 0:
                raise ValueError(
                    "The 99th percentile of 'sct' is negative, leading to a negative scale."
                )
            sct += np.random.normal(
                scale=noise_level * percentile_99 / 10, size=len(sct)
            )

        ut, snt, sct = np.clip(ut, 0, None), np.clip(snt, 0, None), np.clip(sct, 0, None)
        return ut, snt, sct

    def draw_poisson(self, n):
        from random import seed, uniform  # Draw from Poisson

        seed(self.random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # Prepend t0=0

    def is_list(self, x):
        return isinstance(x, (tuple, list, np.ndarray))

    def cycle(self, array, n_vars=None):
        if self.is_list(array):
            return (
                array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
            )
        else:
            return [array] if n_vars is None else [array] * n_vars

    def switch_times(self, t_max, n_vars):
        lower_bound = 0.05 * t_max
        upper_bound = 0.8 * t_max
        uniform_array = np.random.uniform(lower_bound, upper_bound, n_vars)
        return uniform_array

    def simulation(self):
        """
        Runs the simulation with the current parameters.
        """
        n_obs = self.n_obs
        n_vars = self.n_vars
        alpha = self.alpha
        beta = self.beta
        nu = self.nu
        gamma = self.gamma
        alpha_ = self.alpha_
        t_max = self.t_max
        noise_model = self.noise_model
        noise_level = self.noise_level
        random_seed = self.random_seed

        np.random.seed(random_seed)

        t = self.draw_poisson(n_obs)
        if t_max is not None:
            t *= t_max / np.max(t)
        t_max = np.max(t)

        if self.generate_switch_times:
            switches = self.switch_times(t_max = t_max, n_vars = n_vars)
            switches = self.cycle(switches,n_vars)
        else:
            switches = self.cycle([0.4, 0.7, 1, 0.1], n_vars)

        # switches = (
        #     self.cycle([0.4, 0.7, 1, 0.1], n_vars)
        #     if switches is None
        #     else self.cycle(switches, n_vars)
        # )

        # t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])

        t_ = np.array([np.max(t[t < t_i]) for t_i in switches])

        noise_level = self.cycle(
            noise_level, len(switches) if n_vars is None else n_vars
        )

        n_vars = min(len(switches), len(noise_level)) if n_vars is None else n_vars
        U = np.zeros(shape=(len(t), n_vars))
        Sn = np.zeros(shape=(len(t), n_vars))
        Sc = np.zeros(shape=(len(t), n_vars))

        for i in range(n_vars):
            alpha_i = alpha[i] if self.is_list(alpha) and len(alpha) != n_obs else alpha
            beta_i = beta[i] if self.is_list(beta) and len(beta) != n_obs else beta
            nu_i = nu[i] if self.is_list(nu) and len(nu) != n_obs else nu
            gamma_i = gamma[i] if self.is_list(gamma) and len(gamma) != n_obs else gamma
            tau, alpha_vec, u0_vec, sn0_vec, sc0_vec = self.vectorize(
                t,
                t_[i],
                alpha_i,
                beta_i,
                nu_i,
                gamma_i,
                alpha_=self.alpha_,
                u0=0,
                sn0=0,
                sc0=0,
            )

            U[:, i], Sn[:, i], Sc[:, i] = self.simulate_dynamics(
                tau,
                alpha_vec,
                beta_i,
                nu_i,
                gamma_i,
                u0_vec,
                sn0_vec,
                sc0_vec,
                noise_model,
                noise_level[i],
            )

        if self.is_list(alpha) and len(alpha) == n_obs:
            alpha = np.nan
        if self.is_list(beta) and len(beta) == n_obs:
            beta = np.nan
        if self.is_list(nu) and len(nu) == n_obs:
            nu = np.nan
        if self.is_list(gamma) and len(gamma) == n_obs:
            gamma = np.nan

        obs = {"true_t": t.round(2)}
        var = {
            "true_t_": t_[:n_vars],
            "true_alpha": np.ones(n_vars) * alpha,
            "true_beta": np.ones(n_vars) * beta,
            "true_nu": np.ones(n_vars) * nu,
            "true_gamma": np.ones(n_vars) * gamma,
            "true_scaling": np.ones(n_vars),
        }

        layer_spliced_unspliced = {"unspliced": U, "spliced": Sn + Sc}
        layer_nuc_cyto = {"unspliced": U + Sn, "spliced": Sc}

        adata_s_u = AnnData(
            Sn,
            obs=obs,
            var=var,
            layers=layer_spliced_unspliced,
        )
        adata_n_c = AnnData(
            Sn,
            obs=obs,
            var=var,
            layers=layer_nuc_cyto,
        )

        # Generate filenames
        obs_name = f"{n_obs}obs"
        vars_name = f"{n_vars}genes"

        adata_s_u_filename = f"adata_s_u_{obs_name}_{vars_name}"
        adata_n_c_filename = f"adata_n_c_{obs_name}_{vars_name}"

        # Save if required
        if self.save:
            adata_s_u.write(
                os.path.join(self.saving_path, adata_s_u_filename + ".h5ad")
            )
            adata_n_c.write(
                os.path.join(self.saving_path, adata_n_c_filename + ".h5ad")
            )

        # Return a dictionary with the filenames as keys
        return {
            adata_s_u_filename: adata_s_u,
            adata_n_c_filename: adata_n_c,
        }

    def simulate_multi_obs_var(self, n_obs_list, n_vars_list):
        """
        Simulates multiple datasets with different n_obs and n_vars.
        If n_obs_list and n_vars_list are given as lists, it will create datasets accordingly.
        """
        adata_dict = {}
        # Store original n_obs and n_vars
        original_n_obs = self.n_obs
        original_n_vars = self.n_vars

        for n_obs in n_obs_list:
            for n_vars in n_vars_list:
                # Set n_obs and n_vars for this simulation
                self.n_obs = n_obs
                self.n_vars = n_vars
                # Run simulation
                result = self.simulation()
                # Update the adata_dict with updated keys including n_obs and n_vars
                adata_dict.update(result)

        # Restore original n_obs and n_vars
        self.n_obs = original_n_obs
        self.n_vars = original_n_vars
        return adata_dict



    def sample_parameters(self,correlations, sigmas, means=None, n_samples=1, random_state=None):
        """
        Samples alpha, beta, nu, and gamma parameters from a multivariate log-normal distribution
        using a list of six correlation coefficients.

        Parameters:
        - correlations (list of floats): Correlation coefficients in the following order:
            [rho_alpha_beta, rho_alpha_nu, rho_alpha_gamma, rho_beta_nu, rho_beta_gamma, rho_nu_gamma]
        - sigmas (list of floats): Standard deviations for [log(alpha), log(beta), log(nu), log(gamma)]
        - means (list of floats, optional): Means for [log(alpha), log(beta), log(nu), log(gamma)]
                                        Defaults to [ln(10), ln(1), ln(0.5), ln(0.1)] if not provided.
        - n_samples (int, optional): Number of parameter sets to sample. Default is 1.
        - random_state (int or np.random.Generator, optional): Seed or random number generator for reproducibility.

        Returns:
        - alpha (list of floats): Sampled alpha parameters.
        - beta (list of floats): Sampled beta parameters.
        - nu (list of floats): Sampled nu parameters.
        - gamma (list of floats): Sampled gamma parameters.
        """

        num_params = 4  # alpha, beta, nu, gamma

        # Validate input lengths
        if len(correlations) != 6:
            raise ValueError("Correlations list must have exactly 6 elements.")
        if len(sigmas) != num_params:
            raise ValueError(f"Sigmas list must have exactly {num_params} elements.")

        # Validate correlation values are between -1 and 1
        for idx, corr in enumerate(correlations):
            if not -1 <= corr <= 1:
                raise ValueError(f"Correlation at position {idx} ({corr}) is out of bounds. Must be between -1 and 1.")

        # Set default means if not provided
        if means is None:
            means = [np.log(10), np.log(1), np.log(0.5), np.log(0.1)]
        else:
            if len(means) != num_params:
                raise ValueError(f"Means list must have exactly {num_params} elements.")
        means = np.array(means)

        # Convert sigmas to a NumPy array
        sigmas = np.array(sigmas)

        # Extract individual correlations
        rho_alpha_beta, rho_alpha_nu, rho_alpha_gamma, rho_beta_nu, rho_beta_gamma, rho_nu_gamma = correlations

        # Construct the full symmetric correlation matrix
        correlation_matrix = np.array([
            [1.0,             rho_alpha_beta,  rho_alpha_nu,   rho_alpha_gamma],
            [rho_alpha_beta,  1.0,             rho_beta_nu,    rho_beta_gamma],
            [rho_alpha_nu,    rho_beta_nu,     1.0,             rho_nu_gamma],
            [rho_alpha_gamma, rho_beta_gamma,  rho_nu_gamma,    1.0]
        ])

        

        # Check if the correlation matrix is symmetric
        if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8):
            raise ValueError("Constructed correlation matrix is not symmetric.")

        # Construct the covariance matrix
        cov_matrix = correlation_matrix * np.outer(sigmas, sigmas)

        print(f"parameters generated with the covariance matrix: {cov_matrix}")

        # Check if covariance matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues < -1e-8):  # Allow a small negative tolerance due to numerical precision
            raise ValueError("Covariance matrix is not positive semi-definite. Please check your correlations and sigmas.")

        # Set random state for reproducibility if provided
        rng = np.random.default_rng(random_state)

        # Sample from the multivariate normal distribution
        log_params = rng.multivariate_normal(mean=means, cov=cov_matrix, size=n_samples)

        # Exponentiate to get the original parameters
        params = np.exp(log_params)

        # Split the parameters into separate lists
        alpha = params[:, 0].tolist()
        beta = params[:, 1].tolist()
        nu = params[:, 2].tolist()
        gamma = params[:, 3].tolist()

        return alpha, beta, nu, gamma



