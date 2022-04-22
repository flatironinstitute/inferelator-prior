import numpy as np
import scanpy as sc
import anndata as ad
import itertools

from scipy.sparse import issparse
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr

from inferelator.regression.mi import _make_array_discrete, _make_table, _calc_mi

from sklearn.utils import gen_even_slices
from joblib import Parallel, delayed, effective_n_jobs


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def program_select_mi(data, n_programs=2, mi_bins=10, n_comps=None, normalize=True,
                      layer="X", max_comps=100, comp_var_required=0.0025, n_jobs=-1,
                      verbose=False, use_hvg=False):
    """
    Find a specific number of gene programs based on mutual information between genes
    It is highly advisable to use raw counts as input.
    The risk of information leakage is high otherwise.

    :param data: Expression object
    :type data: ad.AnnData
    :param n_programs: Number of gene programs, defaults to 2
    :type n_programs: int, optional
    :param mi_bins: Number of bins to use when discretizing for MI,
        defaults to 10
    :type mi_bins: int, optional
    :param n_comps: Number of components to use,
        will select based on explained variance if None,
        defaults to None
    :type n_comps: int, optional
    :param normalize: Do normalization and preprocessing,
        defaults to True
    :type normalize: bool, optional
    :param layer: Data layer to use, defaults to "X"
    :type layer: str, optional
    :param max_comps: Maximum number of components to use if selecting based on explained variance,
        defaults to 100
    :type max_comps: int, optional
    :param comp_var_required: Threshold for PC explained variance to use if selecting based on
        explained variance,
        defaults to 0.0025
    :type comp_var_required: float, optional
    :param n_jobs: Number of CPU cores to use for parallelization, defaults to -1 (all cores)
    :type n_jobs: int, optional
    :param verbose: Print status, defaults to False
    :type verbose: bool, optional
    :param use_hvg: Use precalculated HVG genes if normalization=False, defaults to False
    :type use_hvg: bool, optional
    :return: Data object with new attributes:
        .obsm['program_PCs']: Principal component for each program
        .var['leiden']: Leiden cluster ID
        .var['program]: Program ID
        .uns['MI_program']: {
            'leiden_correlation': Absolute value of spearman rho between PC1 of each leiden cluster
            'mutual_information_genes': Gene labels for mutual information matrix
            'mutual_information': Mutual information (bits) between genes
            'cluster_program_map': Dict mapping gene clusters to gene programs
        }
    :rtype: _type_
    """

    #### CREATE A NEW DATA OBJECT FOR THIS ANALYSIS ####

    if layer == 'X':
        d = ad.AnnData(data.X, dtype=float)
    else:
        d = ad.AnnData(data.layers[layer], dtype=float)

    d.layers['counts'] = d.X.copy()
    d.var = data.var.copy()
    n, m = d.X.shape

    #### PREPROCESSING / NORMALIZATION ####

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

    #### CALCULATING MUTUAL INFORMATION & GENE CLUSTERING ####

    vprint(f"Calculating MI for {pca_expr.shape} array", verbose=verbose)
    mutual_info = _mutual_information(pca_expr, mi_bins, n_jobs)

    vprint(f"Calculating k-NN and Leiden for {mutual_info.shape} MI array",
           verbose=verbose)
    d.var['leiden'] = _leiden_cluster(mutual_info,
                                      neighbors_kws={'metric': 'correlation'},
                                      leiden_kws={'random_state': 50})

    _n_l_clusts = d.var['leiden'].nunique()

    vprint(f"Found {_n_l_clusts} unique gene clusters",
           verbose=verbose)

    _cluster_pc1 = np.zeros((d.shape[0], _n_l_clusts), dtype=float)
    for i in range(_n_l_clusts):
        _cluster_pc1[:, i] = _get_pc1(d.layers['counts'][:, d.var['leiden'] == i])

    #### SECOND ROUND OF CLUSTERING TO MERGE GENE CLUSTERS INTO PROGRAMS ####

    _rho_pc1 = np.abs(spearmanr(_cluster_pc1))[0]

    vprint(f"Merging {_n_l_clusts} gene clusters into {n_programs} programs",
           verbose=verbose)

    # Merge clusters based on correlation distance (1 - abs(spearman rho))
    clust_2 = AgglomerativeClustering(n_clusters=n_programs,
                                      affinity='precomputed',
                                      linkage='complete').fit_predict(1 - _rho_pc1)

    clust_map = {str(k): str(clust_2[k]) for k in range(_n_l_clusts)}
    clust_map[str(-1)] = str(-1)

    d.var['program'] = d.var['leiden'].astype(str).map(clust_map)

    #### LOAD FINAL DATA INTO INITIAL DATA OBJECT AND RETURN IT ####

    data.var['leiden'] = str(-1)
    data.var.loc[d.var_names, 'leiden'] = d.var['leiden'].astype(str)
    data.var['program'] = data.var['leiden'].map(clust_map)

    # Calculate PC1 for each program
    data.obsm['program_PCs'] = np.zeros((d.shape[0], n_programs), dtype=float)
    for i in range(n_programs):
        data.obsm['program_PCs'][:, i] = _get_pc1(d.layers['counts'][:, d.var['program'] == str(i)])

    data.uns['MI_program'] = {
        'leiden_correlation': _rho_pc1,
        'mutual_information_genes': d.var_names.values,
        'mutual_information': mutual_info,
        'cluster_program_map': clust_map
    }

    return data


def _get_pc1(data, normalize=True):
    """
    Get the values for PC1

    :param data: Data array or matrix [Obs x Var]
    :type data: np.ndarray, sp.spmatrix
    :param normalize: Normalize depth & log transform, defaults to True
    :type normalize: bool, optional
    :return: PC1 [Obs]
    :rtype: np.ndarray
    """
    _l_ad = ad.AnnData(data.A if issparse(data) else data, dtype=float)

    if normalize:
        sc.pp.normalize_per_cell(_l_ad, min_counts=0)
        sc.pp.log1p(_l_ad)

    sc.pp.pca(_l_ad, n_comps=2)

    return _l_ad.obsm['X_pca'][:, 0]


def _leiden_cluster(array, neighbors_kws=None, leiden_kws=None):

    neighbors_kws = {} if neighbors_kws is None else neighbors_kws
    leiden_kws = {} if leiden_kws is None else leiden_kws

    neighbors_kws['use_rep'] = 'X'

    ad_arr = ad.AnnData(array, dtype=float)
    sc.pp.neighbors(ad_arr, **neighbors_kws)
    sc.tl.leiden(ad_arr, **leiden_kws)

    return ad_arr.obs['leiden'].astype(int).values


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
