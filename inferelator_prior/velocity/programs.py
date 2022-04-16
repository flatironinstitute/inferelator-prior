import tqdm
import numpy as np
import scanpy as sc
import anndata as ad

from scipy.sparse import issparse
from scipy import linalg

import sklearn.decomposition
from sklearn.linear_model import ridge_regression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import delayed
from sklearn.utils import gen_even_slices
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from joblib import parallel_backend as _parallel_backend
from joblib import Parallel, effective_n_jobs

# DEFAULT ALPHA SEARCH SPACE #
# 0 to 200 (LOGSPACE <1 & INCREASING STEPS >1) #
ALPHA = np.concatenate((np.array([0]),
                        np.logspace(-3, 0, 4),
                        np.linspace(2, 10, 5),
                        np.linspace(20, 50, 4),
                        np.array([75, 100, 150, 200]))
                       )

ALPHA_LASSO = np.concatenate((np.array([0]),
                              np.logspace(-4, 0, 9),
                              np.linspace(2, 10, 5)))

def program_select(data, alphas=None, batch_size=None, random_state=50, layer='X',
                   n_components=100, normalize=True, ridge_alpha=0.01, threshold='mse',
                   method='lasso', **kwargs):
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

    method = method.lower()

    if method == 'minibatchsparsepca':
        sklearn_sparse = sklearn.decomposition.MiniBatchSparsePCA
    elif method == 'sparsepca':
        sklearn_sparse = sklearn.decomposition.SparsePCA
    elif method == 'lasso':
        sklearn_sparse = ParallelLasso

    if alphas is None and method == 'lasso':
        alphas = ALPHA_LASSO
    elif alphas is None:
        alphas = ALPHA
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

    if batch_size is None:
        batch_size = min(1000, max(int(n/10), 10))

    # Center means
    m_mean = np.mean(d.X, axis=0)
    d.X = d.X - m_mean[None, :]

    # Calculate baseline for deviance
    pca_obj = sklearn.decomposition.PCA(n_components=n_components)
    d.obsm['X_pca'] = pca_obj.fit_transform(d.X)
    d.varm['PCs'] = pca_obj.components_.T

    del pca_obj

    # Switch order
    d.X = np.asfortranarray(d.X)

    d.obsm['X_from_pca'] = _ridge_rotate(
        d.varm['PCs'].T,
        d.obsm['X_pca'].T,
        ridge_alpha
    )

    results = {
        'alphas': alphas,
        'loadings': [],
        'means': m_mean,
        'full_model_mse': mean_squared_error(d.X, d.obsm['X_from_pca']),
        'mse': np.full(a, fill_value=np.nan, dtype=float),
        'mse_full': np.full(a, fill_value=np.nan, dtype=float),
        'bic': np.full(a, fill_value=np.nan, dtype=float),
        'nnz': np.zeros(a, dtype=int),
        'nnz_genes': np.zeros(a, dtype=int),
        'deviance': np.full(a, fill_value=np.nan, dtype=float)
    }

    models = []

    for i in tqdm.trange(a):

        mbsp = sklearn_sparse(n_components=n_components,
                              n_jobs=-1,
                              alpha=alphas[i],
                              random_state=random_state,
                              ridge_alpha=ridge_alpha,
                              batch_size=batch_size,
                              **kwargs)

        with _parallel_backend("loky", inner_max_num_threads=1):

            # Fit a regularized linear model between projection & expression
            if method == 'lasso':
                deviance = mbsp.fit_transform(d.obsm['X_pca'], d.X)
                
                # Add loadings
                results['loadings'].append(mbsp.components_)

            # Do SparsePCA (regularized SVD) on expression
            # And then use ridge regression to rotate back to expression
            else:
                projected = mbsp.fit_transform(d.X)

                deviance = _ridge_rotate(
                    mbsp.components_,
                    projected.T,
                    ridge_alpha
                )

                # Add loadings
                results['loadings'].append(mbsp.components_.T)

            # Calculate errors
            mse = mean_squared_error(deviance, d.obsm['X_from_pca'])
            mse_full = mean_squared_error(deviance, d.X)

        # Deviance from PCA per gene w/same # comps
        deviance -= d.obsm['X_from_pca']
        deviance **= 2
        deviance = np.sum(deviance)

        nnz_per_gene = np.sum(mbsp.components_ != 0, axis=0)

        # Calculate BIC from deviance
        # n * log(deviance / n) + k * log(n)
        k = np.sum(nnz_per_gene) + 1
        results['bic'][i] = n * np.log(deviance / n) + k * np.log(n)



        # Add summary stats
        results['mse'][i] = mse
        results['mse_full'][i] = mse_full
        results['nnz'][i] = np.sum(nnz_per_gene)
        results['nnz_genes'][i] = np.sum(nnz_per_gene > 0)
        results['deviance'][i] = deviance

        models.append(mbsp)

        if np.sum(nnz_per_gene) == 0:
            break

    # Minimum BIC
    if threshold == 'bic':
        select_alpha = np.nanargmin(results['bic'])

    # Minimum MSE
    elif threshold == 'mse':
        select_alpha = np.nanargmin(results['mse'])

    # Largest Alpha w/90% of genes
    elif threshold == 'genes':
        select_alpha = np.nanargmax(alphas[(results['nnz_genes'] / m) > 0.9])

    results['opt_alpha'] = alphas[select_alpha]

    output_key = layer + "_sparsepca"

    if method == 'lasso':
        data.obsm[output_key] = d.obsm['X_pca'] @ np.linalg.pinv(results['loadings'][select_alpha])
    else:
        data.obsm[output_key] = models[select_alpha].transform(d.X)

    # Pad components with zeros if some genes were filtered during normalization
    if results['loadings'][select_alpha].shape[0] != data.shape[1]:
        for i in range(len(results['loadings'])):
            v_out = np.zeros((data.shape[1], n_components), dtype=float)
            v_out[_keep_gene_mask, :] = results['loadings'][i]
            results['loadings'][i] = v_out

    results['loadings'] = np.array(results['loadings'])

    data.varm[output_key] = results['loadings'][select_alpha].copy()
    data.uns['sparse_pca'] = results

    return data


def _ridge_rotate(comps, data, ridge_alpha=0.01, solver="cholesky"):
    return ridge_regression(
                comps,
                data,
                ridge_alpha,
                solver=solver
            )


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

    def fit(self, X, Y, **kwargs):

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
                    **kwargs,
                )
                for i in slices
            )

            for i, results in zip(slices, views):
                coefs[i, :] = results

        self.components_ = coefs

        return self

    def fit_transform(self, X, Y):

        self.fit(X, Y)
        return self.transform(X)

    def transform(self, X):
        return X @ self.coef_.T


@ignore_warnings(category=ConvergenceWarning)
def _lasso(X, y, warm_start=None, **kwargs):

    kwargs['fit_intercept'] = False

    if kwargs['alpha'] <= 0.1 and 'max_iter' not in kwargs:
        kwargs['max_iter'] = 2500

    return Lasso(**kwargs).fit(X, y).coef_
