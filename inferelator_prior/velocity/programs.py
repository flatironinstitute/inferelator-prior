import tqdm
import numpy as np
import scanpy as sc
import anndata as ad

from scipy.sparse import issparse
from scipy.stats import zscore

import sklearn.decomposition
from sklearn.linear_model import ridge_regression

from joblib import parallel_backend as _parallel_backend

# DEFAULT ALPHA SEARCH SPACE #
# 0 to 200 (LOGSPACE <1 & INCREASING STEPS >1) #
ALPHA = np.concatenate((np.array([0]),
                        np.logspace(-4, 0, 15),
                        np.linspace(2, 20, 11),
                        np.linspace(25, 50, 6),
                        np.array([75, 100, 150, 200]))
                       )

OUTLIER_SQUISH = 10


def sparse_PCA(data, alphas=None, batch_size=None, random_state=50, layer='X',
               n_components=100, normalize=True, **kwargs):
    """
    Calculate a sparse PCA using sklearn MiniBatchSparsePCA for a range of
    alpha hyperparameters.

    :param data: Data object
    :type data: ad.AnnData
    :param alphas: A 1d array of alpha parameters, defaults to None.
        If None is passed, a default search space will be used
    :type alphas: np.ndarray, optional
    :param batch_size: The batch_size for MiniBatchSparsePCA, defaults to None
    :type batch_size: int, optional
    :param random_state: The random state for MiniBatchSparsePCA, defaults to 50
    :type random_state: int, optional
    :param layer: Data object layer to use, defaults to 'X'
    :type layer: str, optional
    :param n_components: Number of PCs to evaluate, defaults to 100
    :type n_components: int, optional
    :param normalize: Depth-normalize, log-transform, and scale date before PCA,
        defaults to True
    :type normalize: bool, optional
    :return: Data object with .uns['sparse_pca'], .obsm[], and .varm[] added
    :rtype: ad.AnnData
    """

    if layer == 'X':
        d = ad.AnnData(data.X.astype(float), dtype=float)
    else:
        d = ad.AnnData(data.layers[layer].astype(float), dtype=float)

    if issparse(d.X):
        d.X = d.X.A

    alphas = ALPHA.copy() if alphas is None else alphas

    if normalize:
        # Mask is the same as sc.pp.filter_genes(min_cells=10)
        _keep_gene_mask = np.sum(d.X != 0, axis=0) >= 10

        sc.pp.filter_genes(d, min_cells=10)
        sc.pp.normalize_per_cell(d)
        sc.pp.log1p(d)
        d.X = zscore(d.X)

        ### Squish outliers ###
        d.X[d.X < (-1 * OUTLIER_SQUISH)] = -1 * OUTLIER_SQUISH
        d.X[d.X > OUTLIER_SQUISH] = OUTLIER_SQUISH

    else:
        # Dummy mask
        _keep_gene_mask = np.ones(d.shape[1], dtype=bool)

    if batch_size is None:
        batch_size = max(int(d.shape[0] / 1000), 5)

    n = d.X.shape[0]

    results = {
        'alphas': alphas,
        'loadings': [],
        'mse': np.zeros(alphas.shape, dtype=float),
        'bic': np.zeros((alphas.shape, d.X.shape[1]), dtype=float),
        'bic_joint': np.zeros(alphas.shape, dtype=float),
        'nnz': np.zeros(alphas.shape, dtype=float),
        'nnz_genes': np.zeros(alphas.shape, dtype=float)
    }

    models = []

    for i in tqdm.trange(len(alphas)):

        a = alphas[i]

        mbsp = sklearn.decomposition.MiniBatchSparsePCA(n_components=n_components,
                                                        n_jobs=-1,
                                                        alpha=a,
                                                        batch_size=batch_size,
                                                        random_state=random_state,
                                                        **kwargs)

        with _parallel_backend("loky", inner_max_num_threads=1):
            projected = mbsp.fit_transform(d.X)
            resid = ridge_regression(
                mbsp.components_, projected.T, 0.01, solver="cholesky"
            )

        # MSE per gene
        resid -= d.X
        resid **= 2
        resid = np.mean(resid, axis=0)

        nnz_per_gene = np.sum(mbsp.components_ != 0, axis=0)

        # Calculate BIC from mean squared residuals
        # n * log(MSE) + k * log(n)
        results['bic'][i, :] = n * np.log(resid) + nnz_per_gene * np.log(n)

        # Sum all BIC to get a joint information criterion
        results['bic_joint'] = np.sum(results['bic'][i, :])

        # Add loadings
        results['loadings'].append(mbsp.components_.T)

        # Add summary stats
        results['nnz'][i] = np.sum(mbsp.components_ != 0)
        results['nnz_genes'][i] = np.sum(nnz_per_gene > 0)
        results['mse'][i] = np.mean(resid)

        models.append(mbsp)

    min_mse = np.argmin(results['bic_joint'])

    results['opt_alpha'] = alphas[min_mse]

    output_key = layer + "_sparsepca"

    # Pad components with zeros if some genes were filtered during normalization
    if results['loadings'][min_mse].shape[0] != data.shape[1]:
        for i in range(len(results['loadings'])):
            v_out = np.zeros((data.shape[1], n_components), dtype=float)
            v_out[_keep_gene_mask, :] = results['loadings'][i]
            results['loadings'][i] = v_out

    data.varm[output_key] = results['loadings'][min_mse].copy()
    data.obsm[output_key] = models[min_mse].transform(d.X)
    data.uns['sparse_pca'] = results

    return data
