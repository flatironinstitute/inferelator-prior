import tqdm
import numpy as np
import scanpy as sc
import anndata as ad

from scipy.sparse import issparse
from scipy.stats import zscore

import sklearn.decomposition
from sklearn.linear_model import ridge_regression
from sklearn.metrics import mean_squared_error

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
               n_components=100, normalize=True, ridge_alpha=0.025, threshold='genes',
               minibatchsparsepca=True, **kwargs):
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
    :param threshold: Select optimization threshold, defaults to 'genes'.
        'genes' selects alpha based on retaining 90% of genes in final model
        'mse' minimizes mean squared error against raw data
        'bic' minimizes BIC of deviance from full PCA model
    :type threshold: str, optional
    :param minibatchsparsepca: Use sklearn MiniBatchSparsePCA, defaults to True
    :type minibatchsparsepca: bool, optional
    :param **kwargs: Additional keyword arguments for sklearn.decomposition object
    :return: Data object with .uns['sparse_pca'], .obsm[], and .varm[] added
    :rtype: ad.AnnData
    """

    if layer == 'X':
        d = ad.AnnData(data.X.astype(float), dtype=float)
    else:
        d = ad.AnnData(data.layers[layer].astype(float), dtype=float)

    if issparse(d.X):
        d.X = d.X.A

    if minibatchsparsepca:
        sklearn_sparse = sklearn.decomposition.MiniBatchSparsePCA
    else:
        sklearn_sparse = sklearn.decomposition.SparsePCA

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

    n, m = d.X.shape

    # Calculate baseline for deviance
    with _parallel_backend("loky", inner_max_num_threads=1):
        sc.pp.pca(d, n_comps=n_components, zero_center=True, dtype=float)
        d.obsm['X_from_pca'] = ridge_regression(
            d.varm['PCs'].T,
            d.obsm['X_pca'].T,
            ridge_alpha,
            solver="cholesky"
        )

    results = {
        'alphas': alphas,
        'loadings': [],
        'full_model_mse': mean_squared_error(d.X, d.obsm['X_from_pca']),
        'mse': np.zeros(alphas.shape, dtype=float),
        'bic': np.zeros(alphas.shape, dtype=float),
        'nnz': np.zeros(alphas.shape, dtype=int),
        'nnz_genes': np.zeros(alphas.shape, dtype=int),
        'deviance': np.zeros(alphas.shape, dtype=float)
    }

    models = []

    for i in tqdm.trange(len(alphas)):

        a = alphas[i]

        mbsp = sklearn_sparse(n_components=n_components,
                              n_jobs=-1,
                              alpha=a,
                              random_state=random_state,
                              **kwargs)

        with _parallel_backend("loky", inner_max_num_threads=1):
            projected = mbsp.fit_transform(d.X)
            deviance = ridge_regression(
                mbsp.components_,
                projected.T,
                ridge_alpha,
                solver="cholesky"
            )

        # Cleanup component floats
        comp_eps = np.finfo(mbsp.components_.dtype).eps
        mbsp.components_[np.abs(mbsp.components_) <= comp_eps] = 0.

        # MSE from base data
        mse = mean_squared_error(deviance, d.X)

        # Deviance from PCA per gene
        deviance -= d.obsm['X_from_pca']
        deviance **= 2
        deviance = np.sum(deviance)

        nnz_per_gene = np.sum(mbsp.components_ != 0, axis=0)

        # Calculate BIC from deviance
        # n * log(deviance / n) + k * log(n)
        results['bic'][i] = n * np.log(deviance / n) + np.sum(nnz_per_gene) * np.log(n)

        # Add loadings
        results['loadings'].append(mbsp.components_.T)

        # Add summary stats
        results['mse'][i] = mse
        results['nnz'][i] = np.sum(nnz_per_gene)
        results['nnz_genes'][i] = np.sum(nnz_per_gene > 0)
        results['deviance'][i] = deviance

        models.append(mbsp)

    # Minimum BIC
    if threshold == 'bic':
        select_alpha = np.argmin(results['bic_joint'])

    # Minimum MSE
    elif threshold == 'mse':
        select_alpha = np.argmin(results['mse'])

    # Largest Alpha w/90% of genes
    elif threshold == 'genes':
        select_alpha = np.argmax(alphas[(results['nnz_genes'] / m) > 0.9])

    results['opt_alpha'] = alphas[select_alpha]

    output_key = layer + "_sparsepca"

    # Pad components with zeros if some genes were filtered during normalization
    if results['loadings'][select_alpha].shape[0] != data.shape[1]:
        for i in range(len(results['loadings'])):
            v_out = np.zeros((data.shape[1], n_components), dtype=float)
            v_out[_keep_gene_mask, :] = results['loadings'][i]
            results['loadings'][i] = v_out

    data.varm[output_key] = results['loadings'][select_alpha].copy()
    data.obsm[output_key] = models[select_alpha].transform(d.X)
    data.uns['sparse_pca'] = results

    return data
