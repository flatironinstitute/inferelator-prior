import tqdm
import numpy as np
import scipy.sparse as sps
import scanpy as sc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mcv_pcs(count_data, n=5, n_pcs=100, random_seed=800, p=0.5, metric='mse'):
    """
    Calculate a loss metric based on molecular crossvalidation

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param n: Number of crossvalidation resplits, defaults to 5
    :type n: int, optional
    :param n_pcs: Number of PCs to search, defaults to 100
    :type n_pcs: int, optional
    :param random_seed: Random seed for split, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :param metric: Metric to use - accepts 'mse', 'mae', and 'r2' as strings,
        or any callable of the type metric(pred, true), defaults to 'mse'
    :type metric: str, func, optional
    :return: An n x n_pcs array of metric results
    :rtype: np.ndarray
    """

    if metric == 'mse':
        metric = mean_squared_error
    elif metric == 'mae':
        metric = mean_absolute_error
    elif metric == 'r2':
        metric = r2_score

    metric_arr = np.zeros((n, n_pcs + 1), dtype=float)

    # Use a single progress bar for nested loop
    with tqdm.tqdm(total=n * (n_pcs + 1)) as pbar:

        for i in range(n):
            A, B = _molecular_split(
                count_data,
                random_seed=random_seed,
                p=p
            )

            A = _normalize_for_pca(A)
            B = _normalize_for_pca(B)

            # Densify B no matter what
            # So metric doesn't complain
            if sps.issparse(B):
                B = B.A

            # Calculate PCA
            A_obsm, A_varm, _, _ = sc.pp.pca(
                A,
                n_comps=n_pcs,
                zero_center=True,
                return_info=True
            )

            # Null model (no PCs)
            metric_arr[i, 0] = np.sum(B ** 2)
            pbar.update(1)

            # Calculate metric for 1-n_pcs number of PCs
            for j in range(1, n_pcs + 1):
                metric_arr[i, j] = metric(
                    B,
                    A_obsm[:, 0:j] @ A_varm[0:j, :]
                )
                pbar.update(1)

    return metric_arr


def _normalize_for_pca(count_data, copy=True):
    """
    Depth normalize and log pseudocount

    :param count_data: _description_
    :type count_data: _type_
    :return: _description_
    :rtype: _type_
    """

    return sc.pp.log1p(
        sc.pp.normalize_per_cell(
            count_data.astype(float),
            min_counts=0,
            copy=copy
        )
    )


def _molecular_split(count_data, random_seed=800, p=0.5):
    """
    Break an integer count matrix into two count matrices.
    These will sum to the original count matrix and are
    selected randomly from the binomial distribution

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param random_seed: Random seed for generator, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :return: Two count matrices A & B of the same type as the input count_data,
        where A + B = count_data
    :rtype: np.ndarray or sp.sparse.csr_matrix or sp.sparse.csc_matrix
    """

    rng = np.random.default_rng(random_seed)

    if sps.issparse(count_data):

        mat_func = sps.csr_matrix if sps.isspmatrix_csr else sps.csc_matrix

        cv_data = mat_func((
            rng.binomial(count_data.data, p=p),
            count_data.indices,
            count_data.indptr),
            shape = count_data.shape
        )

        count_data = mat_func((
            count_data.data - cv_data.data,
            count_data.indices,
            count_data.indptr),
            shape = count_data.shape
        )

    else:

        cv_data = np.zeros_like(count_data)

        for i in range(count_data.shape[0]):
            cv_data[i, :] = rng.binomial(count_data[i, :], p=p)

        count_data = count_data - cv_data

    return count_data, cv_data
