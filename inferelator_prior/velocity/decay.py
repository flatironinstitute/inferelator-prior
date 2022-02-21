import numpy as np
from tqdm import trange


def calc_decay(expression_data, velocity_data, include_alpha=True,
               decay_quantiles=(0.00, 0.025), alpha_quantile=0.975):
    """
    Estimate decay constant lambda and for dX/dt = -lambda X + alpha

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
    :raises ValueError: Raises a ValueError if arguments are invalid
    :return: Returns estimates for lambda [M,],
        standard error of lambda estimate [M,],
        and estimates of alpha [M,]
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """

    if expression_data.shape != velocity_data.shape:
        raise ValueError(f"Expression data {expression_data.shape} ",
                         f"and velocity data {velocity_data.shape} ",
                         "are not the same size")

    if ((len(decay_quantiles) != 2) or not isinstance(decay_quantiles, (tuple, list))):
        raise ValueError(f"decay_quantiles must be a tuple of two floats; {decay_quantiles} passed")

    n, m = expression_data.shape

    # Get the velocity / expression ratio
    # Set to 0 where expression is zero
    ratio_data = np.zeros_like(velocity_data, dtype=float)
    np.divide(velocity_data, expression_data, out=ratio_data, where=expression_data != 0)

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
        sl, ssr, rank, s = np.linalg.lstsq(x, y, rcond=None)
        return sl[0]

    # Estimate lambda_hat via OLS slope and enforce positive lambda
    decay_est = np.array([_lstsq(expr[i, keep_observations[i, :]].reshape(-1, 1),
                                 velo[i, keep_observations[i, :]].reshape(-1, 1))
                          for i in trange(m)])
    decay_est *= -1
    np.maximum(decay_est, 0, out=decay_est)

    # Estimate standard error of lambda_hat
    se_est = np.array([_calc_se(expr[i, keep_observations[i, :]].reshape(-1, 1),
                                velo[i, keep_observations[i, :]].reshape(-1, 1),
                                decay_est[i])
                       for i in trange(m)])

    return decay_est, se_est, alpha_est


def _calc_se(x, y, slope):

    mse_x = np.sum(np.square(x - np.mean(x)))
    if mse_x == 0:
        return 0

    elif slope == 0:
        return np.mean(np.square(y - np.mean(y))) / mse_x

    else:
        mse_y = np.sum(np.square(y - np.dot(x, slope)))
        se_y = mse_y / (len(y) - 1)
        return se_y / mse_x
