import tqdm
import numpy as np
import scanpy as sc
import anndata as ad

from scipy.sparse import issparse
from scipy import linalg

import sklearn.decomposition
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import delayed
from sklearn.utils import gen_even_slices

from joblib import parallel_backend as _parallel_backend
from joblib import Parallel, effective_n_jobs

# DEFAULT ALPHA SEARCH SPACE #
# 0 to 10 (LOGSPACE <1 & INCREASING STEPS >1) #
ALPHA_LASSO = np.concatenate((np.array([0]),
                              np.logspace(-4, 0, 17),
                              np.linspace(2, 10, 5)))

def program_select(data, alphas=None, random_state=50, layer='X',
                   n_components=100, normalize=True, threshold='bic',
                   n_jobs=-1, **kwargs):
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

    n, m = d.X.shape

    if alphas is None:
        alphas = ALPHA_LASSO
    else:
        alphas = np.sort(alphas)

    a = alphas.shape[0]

    if normalize:
        # Mask is the same as sc.pp.filter_genes(min_cells=10)
        _keep_gene_mask = np.sum(d.X != 0, axis=0) >= 10

        sc.pp.filter_genes(d, min_cells=10)
        sc.pp.normalize_per_cell(d)
        sc.pp.log1p(d)

    else:
        # Dummy mask
        _keep_gene_mask = np.ones(m, dtype=bool)

    # Center means
    m_mean = np.mean(d.X, axis=0)
    d.X = d.X - m_mean[None, :]

    # Calculate baseline for deviance
    pca_obj = sklearn.decomposition.PCA(n_components=n_components)
    d.obsm['X_pca'] = pca_obj.fit_transform(d.X)
    d.varm['PCs'] = pca_obj.components_.T
    d.obsm['X_from_pca'] = pca_obj.inverse_transform(d.obsm['X_pca'])

    # Switch order
    d.X = np.asfortranarray(d.X)

    results = {
        'alphas': alphas,
        'loadings': [],
        'means': m_mean,
        'full_model_mse': mean_squared_error(d.X, d.obsm['X_from_pca']),
        'mse': np.full(a, fill_value=np.nan, dtype=float),
        'mse_X': np.full(a, fill_value=np.nan, dtype=float),
        'bic': np.full(a, fill_value=np.nan, dtype=float),
        'bic_X': np.full(a, fill_value=np.nan, dtype=float),
        'aic': np.full(a, fill_value=np.nan, dtype=float),
        'aic_X': np.full(a, fill_value=np.nan, dtype=float),
        'nnz': np.zeros(a, dtype=int),
        'nnz_genes': np.zeros(a, dtype=int),
        'deviance': np.full(a, fill_value=np.nan, dtype=float)
    }

    models = []

    for i in tqdm.trange(a):

        mbsp = ParallelLasso(n_components=n_components,
                             n_jobs=n_jobs,
                             alpha=alphas[i],
                             random_state=random_state,
                             **kwargs)

        with _parallel_backend("loky", inner_max_num_threads=1):
            _warm = None if i == 0 else results['loadings'][-1]
            fit_proj = mbsp.fit_transform(d.X,
                                          d.obsm['X_pca'],
                                          warm_start=_warm)

        # Append coefficients [Comps x Genes]
        results['loadings'].append(mbsp.components_)

        # SSR
        ssr = np.sum((fit_proj - d.obsm['X_pca']) ** 2)
        ssr_X = np.sum((pca_obj.inverse_transform(fit_proj) - d.X) ** 2)

        nnz_per_gene = np.sum(mbsp.components_ != 0, axis=0)

        # Calculate BIC from SSR
        # n * log(SSR / n) + k * log(n)
        results['bic'][i] = n * np.log(ssr / n) + np.sum(nnz_per_gene) * np.log(n)
        results['bic_X'][i] = n * np.log(ssr_X / n) + np.sum(nnz_per_gene) * np.log(n)

        results['aic'][i] = n * np.log(ssr / n) + 2 * np.sum(nnz_per_gene)
        results['aic_X'][i] = n * np.log(ssr_X / n) + 2 * np.sum(nnz_per_gene)

        # Add summary stats
        results['mse'][i] = ssr / d.obsm['X_pca'].size
        results['mse_X'][i] = ssr_X / d.X.size
        results['nnz'][i] = np.sum(nnz_per_gene)
        results['nnz_genes'][i] = np.sum(nnz_per_gene > 0)
        results['deviance'][i] = ssr

        models.append(mbsp)

        if np.sum(nnz_per_gene) == 0:
            break

    # Largest Alpha w/90% of genes
    if threshold == 'genes':
        select_alpha = np.nanargmax(alphas[(results['nnz_genes'] / m) > 0.9])
    elif threshold in results:
        select_alpha = np.nanargmin(results[threshold])
    else:
        _msg = f"threshold={threshold} is not a valid argument"
        raise ValueError(_msg)

    results['opt_alpha'] = alphas[select_alpha]

    output_key = layer + "_sparsepca"

    # Pad components with zeros if some genes were filtered during normalization
    if results['loadings'][select_alpha].shape[0] != data.shape[1]:
        for i in range(len(results['loadings'])):
            v_out = np.zeros((data.shape[1], n_components), dtype=float)
            v_out[_keep_gene_mask, :] = results['loadings'][i].T
            results['loadings'][i] = v_out

    results['loadings'] = np.array(results['loadings'])

    data.obsm[output_key] = models[select_alpha].transform(d.X)
    data.varm[output_key] = results['loadings'][select_alpha].copy()
    data.uns['sparse_pca'] = results

    return data


class ParallelLasso:

    alpha = 1.0
    n_jobs = -1
    ridge_alpha = 0.01

    components_ = None

    @property
    def coef_(self):
        return self.components_

    def __init__(self, alpha=1.0, n_jobs=-1, ridge_alpha=0.01, **kwargs):
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.ridge_alpha = ridge_alpha

    def fit(self, X, Y, warm_start=None, **kwargs):

        n, m = X.shape
        p = Y.shape[1]

        coefs = np.zeros((p, m), dtype=float)

        if self.alpha == 0:

            coefs[:] = linalg.solve(np.dot(X.T, X),
                                    np.dot(X.T, Y),
                                    assume_a='sym').T

        elif self.n_jobs == 1:
            coefs[:] = _lasso(X, Y, alpha=self.alpha)

        else:
            gram = np.dot(X.T, X)

            slices = list(gen_even_slices(p, effective_n_jobs(self.n_jobs)))

            views = Parallel(n_jobs=self.n_jobs)(
                delayed(_lasso)(
                    X,
                    Y[:, i],
                    alpha=self.alpha,
                    precompute=gram,
                    warm_start=warm_start[i, :] if warm_start is not None else None
                    **kwargs,
                )
                for i in slices
            )

            for i, results in zip(slices, views):
                coefs[i, :] = results

        self.components_ = coefs

        return self

    def fit_transform(self, X, Y, **kwargs):

        self.fit(X, Y, **kwargs)
        return self.transform(X)

    def transform(self, X):
        return X @ self.coef_.T


def _lasso(X, y, warm_start=None, **kwargs):

    kwargs['fit_intercept'] = False

    if kwargs['alpha'] <= 0.1 and 'max_iter' not in kwargs:
        kwargs['max_iter'] = 2500

    if warm_start is not None:
        lasso_obj = Lasso(warm_start=True, **kwargs)
        lasso_obj.coef_ = warm_start.copy()
        return lasso_obj.fit(X, y).coef_
    else:
        return Lasso(**kwargs).fit(X, y).coef_
