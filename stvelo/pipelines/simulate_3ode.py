
import numpy as np
import warnings
from anndata import AnnData


def unspliced(tau, u0, alpha, beta):
    """TODO."""
    expu = np.exp(-beta * tau)
    return (u0 * expu) + ((alpha / beta) * (1 - expu))


def spliced_nucleus(tau, sn0, alpha, nu, beta, u0):
    term1 = sn0 * np.exp(-nu * tau)
    term2 = (alpha / nu) * (1 - np.exp(-nu * tau))
    term3 =( (alpha - beta * u0) / (nu - beta) )* (np.exp(-nu * tau) - np.exp(-beta * tau))
    
    return term1 + term2 + term3


def spliced_cyto(tau, alpha, beta, gamma, nu, u0, sn0, sc0):
    
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



def vectorize(t, t_, alpha, beta, nu, gamma=None, alpha_=0, u0=0, sn0=0, sc0=0, sorted=False):
    
    """
    Vectorizes the parameters for mRNA splicing kinetics simulation based on the given time points.

    This function computes the vectorized values of the parameters `tau`, `alpha`, `u0`, and `s0`
    for a set of time points `t` and a threshold time `t_`.

    Parameters
    ----------
    t : array_like
        Array of time points at which to compute the parameters.
    t_ : float
        Threshold time point which separates two different regimes in the simulation.
    alpha : array_like or float
        Transcription rate parameter. This can be a scalar or an array of the same length as `t`.
    beta : array_like or float
        Splicing rate parameter. This can be a scalar or an array of the same length as `t`.
    gamma : array_like or float, optional
        Degradation rate parameter. Defaults to `beta / 2` if not provided.
    alpha_ : float, optional
        Secondary transcription rate parameter used after the threshold time `t_`. Defaults to 0.
    u0 : float, optional
        Initial unspliced mRNA level. Defaults to 0.
    s0 : float, optional
        Initial spliced mRNA level. Defaults to 0.
    sorted : bool, optional
        If True, the resulting arrays are sorted by time `t`. Defaults to False.

    Returns

    tuple of numpy.ndarray
        Tuple containing the following elements:
        - `tau` : numpy.ndarray
            Vectorized time points after applying the threshold `t_`.
        - `alpha` : numpy.ndarray
            Vectorized transcription rate parameter.
        - `u0` : numpy.ndarray
            Vectorized initial unspliced mRNA levels.
        - `s0` : numpy.ndarray
            Vectorized initial spliced mRNA levels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)

    sn0_ = spliced_nucleus(t_, sn0, alpha, nu, beta, u0)

    sc0_ = spliced_cyto(t_, alpha, beta, gamma, nu, u0, sn0, sc0)


    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    sn0 = sn0 * o + sn0_ * (1 - o)
    sc0 = sc0 * o + sc0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, sn0, sc0


def simulation_3ode(
    n_obs=300,
    n_vars=None,
    alpha=None,
    beta=None,
    nu = None,
    gamma=None,
    alpha_=None,
    t_max=None,
    noise_model="normal",
    noise_level=1,
    switches=None,
    random_seed=0,
):
    """Simulation of mRNA splicing kinetics.

    Simulated mRNA metabolism with transcription, splicing and degradation.
    The parameters for each reaction are randomly sampled from a log-normal distribution
    and time events follow the Poisson law. The total time spent in a transcriptional
    state is varied between two and ten hours.

    .. image:: https://user-images.githubusercontent.com/31883718/79432471-16c0a000-7fcc-11ea-8d62-6971bcf4181a.png
       :width: 600px

    Returns
    -------
    Returns tuple `adata_s_u, adata_n_c`
    
    """
    import numpy as np
    np.random.seed(random_seed)

    def draw_poisson(n):
        from random import seed, uniform  # draw from poisson

        seed(random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # prepend t0=0

    def simulate_dynamics(tau, alpha, beta, nu, gamma, u0, sn0, sc0, noise_model, noise_level):
        ut = unspliced(tau, u0, alpha, beta)
        snt = spliced_nucleus(tau, sn0, alpha, nu, beta, u0)
        sct = spliced_cyto(tau, alpha, beta, gamma, nu, u0, sn0, sc0)

        if noise_model == "normal":  # add noise
            ut += np.random.normal(
                scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
            )
            snt += np.random.normal(
                scale=noise_level * np.percentile(snt, 99) / 10, size=len(snt)
            )

            percentile_99 = np.percentile(sct, 99)
            
            if percentile_99 < 0:
                raise ValueError("The 99th percentile of 'sct' is negative, leading to a negative scale.")
            
            sct += np.random.normal(
                scale=noise_level * np.percentile(sct, 99) / 10, size=len(sct)
            )
        ut, snt, sct = np.clip(ut, 0, None), np.clip(snt, 0, None), np.clip(sct, 0, None)
        return ut, snt, sct

    #(4,0.6,0.3,0.25)
    alpha = 5 if alpha is None else alpha
    beta = 0.6 if beta is None else beta
    nu = 0.3 if nu is None else nu
    gamma = 0.25 if gamma is None else gamma
    alpha_ = 0 if alpha_ is None else alpha_

    t = draw_poisson(n_obs)
    if t_max is not None:
        t *= t_max / np.max(t)
    t_max = np.max(t)

    def cycle(array, n_vars=None):
        if isinstance(array, (np.ndarray, list, tuple)):
            return (
                array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
            )
        else:
            return [array] if n_vars is None else [array] * n_vars
    

    def switch_times(t_max, n_vars):
        lower_bound = 0.1 * t_max
        upper_bound = 0.5 * t_max
    
        uniform_array = np.random.uniform(lower_bound, upper_bound, n_vars)
    
        return uniform_array


    # switching time point obtained as fraction of t_max rounded down

    switches = (
        cycle([0.4, 0.7, 1, 0.1], n_vars)
        # cycle([4, 5, 6, 8], n_vars)
        if switches is None
        else cycle(switches, n_vars)
    )

    # switches = switch_times(t_max=t_max,n_vars=n_vars)
    # print(switches)

    t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])
    # t_ = switches

    noise_level = cycle(noise_level, len(switches) if n_vars is None else n_vars)

    n_vars = min(len(switches), len(noise_level)) if n_vars is None else n_vars
    U = np.zeros(shape=(len(t), n_vars))
    Sn = np.zeros(shape=(len(t), n_vars))
    Sc = np.zeros(shape=(len(t), n_vars))

    def is_list(x):
        return isinstance(x, (tuple, list, np.ndarray))

    for i in range(n_vars):
        alpha_i = alpha[i] if is_list(alpha) and len(alpha) != n_obs else alpha
        beta_i = beta[i] if is_list(beta) and len(beta) != n_obs else beta
        nu_i = nu[i] if is_list(nu) and len(nu) != n_obs else nu
        gamma_i = gamma[i] if is_list(gamma) and len(gamma) != n_obs else gamma
        tau, alpha_vec, u0_vec, sn0_vec, sc0_vec = vectorize(
            t, t_[i], alpha_i, beta_i,nu_i, gamma_i, alpha_=alpha_, u0=0, sn0=0,sc0=0
        )


        U[:, i], Sn[:, i], Sc[:, i] = simulate_dynamics(
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

    if is_list(alpha) and len(alpha) == n_obs:
        alpha = np.nan
    if is_list(beta) and len(beta) == n_obs:
        beta = np.nan
    if is_list(nu) and len(nu) == n_obs:
        nu = np.nan
    if is_list(gamma) and len(gamma) == n_obs:
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

    # layers = {"unspliced": U,"spliced":Sn+Sc, "spliced_nuc": Sn, "spliced_cyto": Sc, "nucleic": U+Sn}
    layer_spliced_unspliced = {"unspliced": U, "spliced":Sn+Sc}
    layer_nuc_cyto = {"unspliced": U+Sn, "spliced": Sc}

    
    # return AnnData(Sn, obs, var, layers=layers)
    return {'adata_s_u':AnnData(Sn,obs,var,layers=layer_spliced_unspliced), 'adata_n_c':AnnData(Sn,obs, var, layers=layer_nuc_cyto)}
    # return AnnData(Sn,obs,var,layers=layer_spliced_unspliced), AnnData(Sn,obs, var, layers=layer_nuc_cyto)