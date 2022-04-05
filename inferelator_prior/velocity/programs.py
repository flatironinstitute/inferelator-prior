import tqdm
import numpy as np

import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from joblib import parallel_backend as _parallel_backend

ALPHA = np.concatenate((np.array([0]),
                        np.logspace(-4, 0, 15),
                        np.linspace(2, 20, 11),
                        np.linspace(25, 50, 6),
                        np.array([75, 100, 150, 200])
                       ))


def sparse_PCA(data, alphas=None, batch_size=None, random_state=50, layer='X',
               n_components=100):

    d = data.X if layer == 'X' else data.layers[layer]
    alphas = ALPHA.copy() if alphas is None else alphas

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
            projected = mbsp.fit_transform(d)
            back_rotate = (np.linalg.pinv(mbsp.components_) @ projected.T).T

        results['loadings'].append(mbsp.components_.T)
        results['nnz'][i] = np.sum(mbsp.components_ != 0)
        results['nnz_genes'][i] = np.sum(np.sum(mbsp.components_ != 0, axis=0) > 0)
        results['mse'][i] = mean_squared_error(back_rotate, d)

        models.append(mbsp)


    min_mse = np.argmin(results['mse'])

    results['opt_alpha'] = alphas[min_mse]
    data.uns['sparse_pca'] = results

    output_key = layer + "_sparsepca"
    data.varm[output_key] = results['loadings'][str(alphas[min_mse])].copy()
    data.obsm[output_key] = mbsp[min_mse].transform(d)

    return data
