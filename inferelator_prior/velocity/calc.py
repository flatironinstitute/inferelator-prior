import numpy as _np
from scipy.sparse import issparse as _is_sparse
from tqdm import trange

def calc_velocity(expr, time_axis, neighbor_graph, n_neighbors, wrap_time=None):
    """
    Calculate local RNA velocity

    :param expr: Samples x Genes numpy with expression data
    :param time_axis: Samples, numpy array
    :param neighbor_graph: Samples x Samples numpy or scipy.sparse with nearest neighbor distances
    :param n_neighbors: Number of neighbors to use, int
    :return: Samples x Genes numpy with velocity data
    """

    n_gen = _find_local(expr, neighbor_graph, n_neighbors)
    return _np.vstack([_calc_local_velocity(expr[n_idx, :].copy(),
                                            time_axis[n_idx].copy(),
                                            (n_idx == i).nonzero()[0][0],
                                            wrap_time=wrap_time)
                       for i, n_idx in n_gen])


def _calc_local_velocity(expr, time_axis, center_index, wrap_time=None):
    """
    Calculate a local rate of change

    :param expr: Samples x Genes numpy with expression data
    :param time_axis: Samples, numpy array
    :param center_index: The data point which we're calculating velocity for
    :return:
    """

    n, m = expr.shape

    if _np.isnan(time_axis[center_index]):
        return _np.full(m, _np.nan)

    if wrap_time is not None:

        wtime_l, wtime_r = wrap_time * 0.25, wrap_time * 0.75

        # Calculate change in time relative to the centerpoint
        if time_axis[center_index] > wtime_r:
            time_axis[time_axis < wtime_l] = time_axis[time_axis < wtime_l] + wrap_time
        elif time_axis[center_index] < wtime_l:
            time_axis[time_axis > wtime_r] = time_axis[time_axis > wtime_r] - wrap_time

    # Calculate change in expression and time relative to the centerpoint
    y_diff = _np.subtract(expr, expr[center_index, :])
    time_axis = (time_axis - time_axis[center_index])

    _time_nan = _np.isnan(time_axis)

    # Remove any data where the time value is NaN
    if _np.sum(_time_nan) > 0:

        # If there's only 3 or fewer data points remaining, return NaN
        if _np.sum(~_time_nan) < 4:
            return _np.full(m, _np.nan)

        time_axis = time_axis[~_time_nan]
        y_diff = y_diff[~_time_nan, :]

    # Calculate (XT * X)^-1 * XT
    time_axis = time_axis.reshape(-1, 1)
    x_for_hat = _np.dot(_np.linalg.inv(_np.dot(time_axis.T, time_axis)), time_axis.T)

    # Return the slope for each gene as velocity
    return _np.array([_np.dot(x_for_hat, y_diff[:, i])[0] for i in range(m)])


def _find_local(expr, neighbor_graph, n_neighbors):
    """
    Find a return an expression matrix for a locally connected graph

    :param expr: Samples x Genes numpy or scipy with expression data
    :param neighbor_graph: Samples x Samples numpy or scipy with neighbor distances as 1/dist.
    :param n_neighbors:
    :return:
    """

    n, m = expr.shape
    neighbor_sparse = _is_sparse(neighbor_graph)

    for i in trange(n):
        n_slice = neighbor_graph[i, :]
        if neighbor_sparse:
            if n_slice.data.shape[0] > n_neighbors:
                keepers = n_slice.indices[_np.argsort(n_slice.data)[-1 * n_neighbors:]]
            else:
                keepers = n_slice.indices
        else:
            keepers = _np.argsort(n_slice)[-1 * n_neighbors:]

        yield i, _np.insert(keepers, 0, i)
