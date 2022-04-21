import numpy as np
import scanpy as sc
import anndata as ad
import itertools
from scipy.sparse import issparse

from inferelator.regression.mi import _make_array_discrete, _make_table, _calc_mi

from sklearn.utils import gen_even_slices
from joblib import Parallel, delayed, effective_n_jobs


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def program_select_mi(data, mi_bins=10, n_comps=None, n_clusters=2, normalize=True,
                      layer="X", max_comps=100, comp_var_required=0.0025, n_jobs=-1,
                      verbose=False, use_hvg=False):

    if layer == 'X':
        d = ad.AnnData(data.X.astype(float), dtype=float)
    else:
        d = ad.AnnData(data.layers[layer].astype(float), dtype=float)

    d.var = data.var.copy()

    if issparse(d.X):
        d.X = d.X.A

    n, m = d.X.shape

    # Normalize, if required
    if normalize:
        sc.pp.normalize_per_cell(d)
        sc.pp.log1p(d)
        sc.pp.highly_variable_genes(d, max_mean=np.inf, min_disp=0.01)

        _keep_gene_mask = d.var['highly_variable'].values
        d._inplace_subset_var(_keep_gene_mask)

        vprint(f"Normalized and kept {d.shape[1]} highly variable genes",
               verbose=verbose)

    elif use_hvg:
        _keep_gene_mask = data.var['highly_variable'].values
        d._inplace_subset_var(_keep_gene_mask)
        vprint(f"Using {d.shape[1]} highly variable genes",
               verbose=verbose)

    else:
        # Dummy mask
        _keep_gene_mask = np.ones(m, dtype=bool)
        vprint(f"Skipped normalization and kept {d.shape[1]} genes",
               verbose=verbose)

    # PCA
    if n_comps is None:
        sc.pp.pca(d, n_comps=max_comps)

        # Select comps explaining more than threshold
        n_comps = np.sum(d.uns['pca']['variance_ratio'] >= comp_var_required)
        d.obsm['X_pca'] = d.obsm['X_pca'][:, 0:n_comps]
        d.varm['PCs'] = d.varm['PCs'][:, 0:n_comps]

    else:
        sc.pp.pca(d, n_comps=n_comps)

    vprint(f"Using {n_comps} components", verbose=verbose)

    # Rotate back to expression space
    pca_expr = d.obsm['X_pca'] @ d.varm['PCs'].T
    pca_expr = _make_array_discrete(pca_expr, mi_bins, axis=0)

    vprint(f"Calculating MI for {pca_expr.shape} array", verbose=verbose)
    mutual_info = _mutual_information(pca_expr, mi_bins, n_jobs)

    ad_mi = ad.AnnData(mutual_info, dtype=float, obs=d.var)

    vprint(f"Calculating k-NN and Leiden for {ad_mi.shape} MI array", verbose=verbose)

    sc.pp.neighbors(ad_mi, metric='correlation', use_rep='X')
    sc.tl.leiden(ad_mi, random_state=50)

    vprint(f"Identified {len(ad_mi.obs['leiden'].cat.categories)} unique clusters",
           verbose=verbose)

    return ad_mi


def _mutual_information(discrete_array, bins, n_jobs):

    m, n = discrete_array.shape

    slices = list(gen_even_slices(n, effective_n_jobs(n_jobs)))

    views = Parallel(n_jobs=n_jobs)(
        delayed(_mi_slice)(
            discrete_array,
            discrete_array[:, i],
            bins,
            logtype=np.log2
        )
        for i in slices
    )

    mutual_info = np.empty((n, n), dtype=float)

    for i, r in zip(slices, views):
        mutual_info[:, i] = r

    return mutual_info


def _mi_slice(x, y, bins, logtype=np.log):

    n1, n2 = x.shape[1], y.shape[1]

    mutual_info = np.empty((n1, n2), dtype=float)
    for i, j in itertools.product(range(n1), range(n2)):
        mutual_info[i, j] = _calc_mi(_make_table(x[:, i], y[:, j], bins),
                                     logtype=logtype)

    return mutual_info
