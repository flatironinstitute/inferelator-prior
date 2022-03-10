import numpy as np
import scipy.stats 
from tqdm import trange, tqdm


def calc_decay_sliding_windows(expression_data, velocity_data, time_data, n_windows=100, include_alpha=True, **kwargs):

    n, m = expression_data.shape

    min_time, max_time = np.min(time_data), np.max(time_data)

    half_width = (max_time - min_time) / (2 * n_windows + 1)
    centers = np.linspace(min_time + half_width, max_time - half_width, num=n_windows)

    def _calc_window_decay(center):
        lowend, highend = center - half_width, center + half_width

        keep_idx = (time_data >= lowend) & (time_data <= highend)

        if np.sum(keep_idx) < 2:
            return (np.full(expression_data.shape[1], np.nan),
                    np.full(expression_data.shape[1], np.nan),
                    np.full(expression_data.shape[1], np.nan) if include_alpha else None)

        return calc_decay(expression_data[keep_idx, :],
                          velocity_data[keep_idx, :],
                          lstatus=False,
                          include_alpha=include_alpha,
                          **kwargs)

    results = [_calc_window_decay(x) for x in tqdm(centers)]
    return [x[0] for x in results], [x[1] for x in results], [x[2] for x in results], centers


def calc_decay_bootstraps(expression_data, velocity_data, n_bootstraps=15, bootstrap_ratio=1.0,
                          random_seed=8675309, lstatus=True, confidence_interval = 0.95, 
                          **kwargs):
    """
    Estimate decay constant lambda for dX/dt = -lambda X + alpha and calculate
    confidence intervals by bootstrapping.

    :param expression_data: Gene expression data [N Observations x M Genes]
    :type expression_data: np.ndarray (float)
    :param velocity_data: Gene velocity data [N Observations x M Genes]
    :type velocity_data: np.ndarray (float)
    :param n_bootstraps: Number of bootstraps, defaults to 15
    :type n_bootstraps: int, optional
    :param bootstrap_ratio: Fraction of samples to select for each bootstrap,
        defaults to 1.0
    :type bootstrap_ratio: float, optional
    :param random_seed: Seed for bootstrapping RNG, defaults to 8675309
    :type random_seed: int, optional
    :param lstatus: Display status bar, defaults to True
    :type lstatus: bool, optional
    :param confidence_interval: Confidence interval between 0 and 1, defaults to 0.95
    :type confidence_interval: float, optional
    """

    if n_bootstraps < 2:
        raise ValueError(f'n_bootstraps must be > 1, {n_bootstraps} provided')

    lstatus = trange if lstatus else range
    rng = np.random.RandomState(seed=random_seed)

    # Number to select per bootstrap
    # Minimum of 1
    n_to_choose = max(1, int(bootstrap_ratio * expression_data.shape[0]))

    def _calc_boot():
        pick_idx = rng.choice(np.arange(expression_data.shape[0]), size=n_to_choose)
        return calc_decay(expression_data[pick_idx, :],
                          velocity_data[pick_idx, :],
                          lstatus=False, **kwargs)

    bootstrap_results = [_calc_boot() for _ in range(n_bootstraps)]
    decays = np.vstack([x[0] for x in bootstrap_results])
    alphas = np.vstack([x[2] for x in bootstrap_results]) if bootstrap_results[0][2] is not None else None

    t = scipy.stats.t.ppf((1 + confidence_interval) / 2, n_bootstraps - 1)
    ci = t * np.std(decays, axis=0) / np.sqrt(n_bootstraps)

    return np.mean(decays, axis=0), ci, alphas

def calc_decay(expression_data, velocity_data, include_alpha=True,
               decay_quantiles=(0.00, 0.025), alpha_quantile=0.975,
               add_pseudocount=False, log_expression=False, lstatus=True):
    """
    Estimate decay constant lambda for dX/dt = -lambda X + alpha

    :param expression_data: Gene expression data [N Observations x M Genes]
    :type expression_data: np.ndarray (float)
    :param velocity_data: Gene velocity data [N Observations x M Genes]
    :type velocity_data: np.ndarray (float)
    :param decay_quantiles: The quantile of observations to fit lambda,
        defaults to (0.00, 0.025)
    :type decay_quantiles: tuple, optional
    :param alpha_quantile: The quantile of observations to estimate alpha,
        defaults to 0.975
    :type alpha_quantile: float, optional
    :param add_pseudocount: Add a pseudocount to expression for ratio
        calculation, defaults to False
    :type add_pseudocount: bool, optional
    :param log_expression: Log expression for ratio calculation,
        defaults to False
    :type log_expression: bool, optional
    :param lstatus: Display status bar, defaults to True
    :type lstatus: bool, optional
    :raises ValueError: Raises a ValueError if arguments are invalid
    :return: Returns estimates for lambda [M,],
        standard error of lambda estimate [M,],
        and estimates of alpha [M,]
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """

    lstatus = trange if lstatus else range

    if expression_data.shape != velocity_data.shape:
        raise ValueError(f"Expression data {expression_data.shape} ",
                         f"and velocity data {velocity_data.shape} ",
                         "are not the same size")

    if ((len(decay_quantiles) != 2) or not isinstance(decay_quantiles, (tuple, list))):
        raise ValueError(f"decay_quantiles must be a tuple of two floats; {decay_quantiles} passed")

    n, m = expression_data.shape

    # Get the velocity / expression ratio
    # add_pseudocount and log_expression influence this only
    # not decay calculations later
    if add_pseudocount and log_expression:
        ratio_data = np.full_like(velocity_data, np.nan, dtype=float)
        ratio_data = np.divide(velocity_data, np.log1p(expression_data),
                               out=ratio_data, where=expression_data != 0)
    elif add_pseudocount:
        ratio_data = np.divide(velocity_data, expression_data + 1)
    else:
        ratio_data = np.full_like(velocity_data, np.nan, dtype=float)
        np.divide(velocity_data, np.log(expression_data) if log_expression else expression_data,
                  out=ratio_data, where=expression_data != 0)

    # Find the quantile cutoffs for decay curve fitting
    ratio_cuts = np.nanquantile(ratio_data, decay_quantiles, axis=0)

    # Find the observations which should not be included for decay constant model
    keep_observations = np.greater_equal(ratio_data, ratio_cuts[0, :][None, :])
    keep_observations &= np.less_equal(ratio_data, ratio_cuts[1, :][None, :])

    # Estimate the maximum velocity
    if include_alpha:
        alpha_est = np.nanquantile(velocity_data, alpha_quantile, axis=0).flatten()
        np.maximum(alpha_est, 0, out=alpha_est)
    else:
        alpha_est = None

    # Transpose so gene data is memory-contiguous

    if include_alpha:
        # Remove the maximum velocity from the velocity matrix
        velo = np.subtract(velocity_data.T, alpha_est[:, None])
    else:
        velo = np.array(velocity_data.T, copy=True)


    expr = np.array(expression_data.T, copy=True)
    keep_observations = np.array(keep_observations.T, order="C")

    def _lstsq(x, y):
        if x.shape[0] == 0:
            return 0
        sl, ssr, rank, s = np.linalg.lstsq(x, y, rcond=None)
        return sl[0][0]

    # Estimate lambda_hat via OLS slope and enforce positive lambda
    decay_est = np.array([_lstsq(expr[i, keep_observations[i, :]].reshape(-1, 1),
                                 velo[i, keep_observations[i, :]].reshape(-1, 1))
                          for i in lstatus(m)])

    np.minimum(decay_est, 0, out=decay_est)

    # Estimate standard error of lambda_hat
    se_est = np.array([_calc_se(expr[i, keep_observations[i, :]].reshape(-1, 1),
                                velo[i, keep_observations[i, :]].reshape(-1, 1),
                                decay_est[i])
                       for i in lstatus(m)])

    return decay_est * -1, se_est, alpha_est


def _calc_se(x, y, slope):

    if x.shape[0] == 0:
        return 0

    mse_x = np.sum(np.square(x - np.nanmean(x)))
    if mse_x == 0:
        return 0

    elif slope == 0:
        return np.mean(np.square(y - np.nanmean(y))) / mse_x

    else:
        mse_y = np.sum(np.square(y - np.dot(x, slope)))
        se_y = mse_y / (len(y) - 1)
        return se_y / mse_x
