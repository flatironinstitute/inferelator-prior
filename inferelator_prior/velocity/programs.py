import numpy as np
import scanpy as sc
import anndata as ad
import itertools

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
    data.obsm['program_PCs'] = program_pcs(d.layers['counts'], d.var['program'],
                                           program_id_levels=list(map(str, range(n_programs))),
                                           normalize=False)

    data.uns['MI_program'] = {
        'leiden_correlation': _rho_pc1,
        'mutual_information_genes': d.var_names.values,
        'mutual_information': mutual_info,
        'cluster_program_map': clust_map
    }

    return data


def program_pcs(data, program_id_vector, program_id_levels=None,
                skip_program_ids=(str(-1)), normalize=True):
    """
    Calculate principal components for a set of expression programs

    :param data: Expression data [Obs x Features]
    :type data: np.ndarray, sp.spmatrix
    :param program_id_vector: List mapping Features to Program ID
    :type program_id_vector: pd.Series, np.ndarray, list
    :param skip_program_ids: Program IDs to skip, defaults to (str(-1))
        Ignored if program_id_levels is set.
    :type skip_program_ids: tuple, list, pd.Series, np.ndarray, optional
    :param program_id_levels: Program IDs and order for output array
    :type program_id_levels: pd.Series, np.ndarray, list, optional
    :param normalize: Normalize expression data, defaults to False
    :type normalize: bool, optional
    :returns: A numpy array with the program PC1 vector for each program,
        and a list of Program IDs if program_id_levels is not set
    :rtype: np.ndarray, list (optional)
    """

    if program_id_levels is None:
        use_ids = [i for i in np.unique(program_id_vector) if i not in skip_program_ids]
    else:
        use_ids = program_id_levels

    p_pcs = np.zeros((data.shape[0], len(use_ids)), dtype=float)
    for i, prog_id in enumerate(use_ids):
        p_pcs[:, i] = _get_pc1(data[:, program_id_vector == prog_id],
                               normalize=normalize)

    if program_id_levels is not None:
        return p_pcs

    else:
        return p_pcs, use_ids


def information_distance(discrete_array, bins, n_jobs=-1, logtype=np.log, return_information=False):

    # Calculate MI(X, X)
    mi_xx = _mutual_information(discrete_array, bins, logtype=logtype,
                                n_jobs=n_jobs)

    # Calculate H(X)
    h_x = _shannon_entropy(discrete_array, bins, logtype=logtype,
                           n_jobs=n_jobs)

    # Calulate distance as 1 - MI(X, X) / H(X, X)
    # Where H(X, X) = H(X) + H(X) - MI(X, X)
    d_xx = 1 - mi_xx / (h_x[None, :] + h_x[:, None] - mi_xx)

    # Trim floats to 0 based on machine precision
    # Might need a looser tol; there's a lot of float ops here
    d_xx[np.abs(d_xx) <= (bins * np.spacing(bins))] = 0.

    # Return distance or distance & MI
    if return_information:
        return d_xx, mi_xx
    else:
        return d_xx


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
    _l_ad = ad.AnnData(data, dtype=float)

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


def _mutual_information(discrete_array, bins, n_jobs=-1, logtype=np.log):
    """
    Calculate mutual information between features of a discrete array

    :param discrete_array: Discrete integer array with values from 0 to `bins`
    :type discrete_array: np.ndarray [int]
    :param bins: Number of discrete bins in integer array
    :type bins: int
    :param n_jobs: Number of parallel jobs for joblib,
        -1 uses all cores
        None does not parallelize
    :type n_jobs: int, None
    :return: Mutual information array in bits
    :rtype: np.ndarray [float]
    """

    m, n = discrete_array.shape

    slices = list(gen_even_slices(n, effective_n_jobs(n_jobs)))

    views = Parallel(n_jobs=n_jobs)(
        delayed(_mi_slice)(
            discrete_array,
            discrete_array[:, i],
            bins,
            logtype=logtype
        )
        for i in slices
    )

    mutual_info = np.empty((n, n), dtype=float)

    for i, r in zip(slices, views):
        mutual_info[:, i] = r

    return mutual_info


def _shannon_entropy(discrete_array, bins, n_jobs=-1, logtype=np.log):

    m, n = discrete_array.shape

    slices = list(gen_even_slices(n, effective_n_jobs(n_jobs)))

    views = Parallel(n_jobs=n_jobs)(
        delayed(_entropy_slice)(
            discrete_array[:, i],
            bins,
            logtype=logtype
        )
        for i in slices
    )

    entropy = np.empty(n, dtype=float)

    for i, r in zip(slices, views):
        entropy[i] = r

    return entropy


def _entropy_slice(x, bins, logtype=np.log):

    def _entropy(vec):
        px = np.bincount(vec, minlength=bins) / vec.size
        return -1 * np.nansum(px * logtype(px))

    return np.apply_along_axis(_entropy, 0, x)


def _mi_slice(x, y, bins, logtype=np.log):

    n1, n2 = x.shape[1], y.shape[1]

    mutual_info = np.empty((n1, n2), dtype=float)
    for i, j in itertools.product(range(n1), range(n2)):
        mutual_info[i, j] = _calc_mi(_make_table(x[:, i], y[:, j], bins),
                                     logtype=logtype)

    return mutual_info
