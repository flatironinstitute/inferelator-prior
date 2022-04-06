import tqdm
import numpy as np
import scanpy as sc
import anndata as ad

from scipy.sparse import issparse
from scipy.stats import zscore

import sklearn.decomposition
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


def sparse_PCA(data, alphas=None, batch_size=None, random_state=50, layer='X',
               n_components=100, normalize=True):

    if layer == 'X':
        d = ad.AnnData(data.X.astype(float), dtype=float)
    else:
        d = ad.AnnData(data.layers[layer].astype(float), dtype=float)

    if issparse(d.X):
        d.X = d.X.A

    alphas = ALPHA.copy() if alphas is None else alphas

    if normalize:
        sc.pp.normalize_per_cell(d)
        sc.pp.log1p(d)
        sc.pp.filter_genes(d, min_cells=10)
        d.X = zscore(d.X)

    if batch_size is None:
        batch_size = max(int(d.shape[0] / 1000), 5)

    results = {
        'alphas': alphas,
        'loadings': [],
        'mse': np.zeros(alphas.shape, dtype=float),
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
                                                        random_state=random_state)

        with _parallel_backend("loky", inner_max_num_threads=1):
            projected = mbsp.fit_transform(d.X)
            back_rotate = (np.linalg.pinv(mbsp.components_) @ projected.T).T

        results['loadings'].append(mbsp.components_.T)
        results['nnz'][i] = np.sum(mbsp.components_ != 0)
        results['nnz_genes'][i] = np.sum(np.sum(mbsp.components_ != 0, axis=0) > 0)
        results['mse'][i] = mean_squared_error(back_rotate, d.X)

        models.append(mbsp)


    min_mse = np.argmin(results['mse'])

    results['opt_alpha'] = alphas[min_mse]
    data.uns['sparse_pca'] = results

    output_key = layer + "_sparsepca"
    data.varm[output_key] = results['loadings'][str(alphas[min_mse])].copy()
    data.obsm[output_key] = mbsp[min_mse].transform(d.X)

    return data

