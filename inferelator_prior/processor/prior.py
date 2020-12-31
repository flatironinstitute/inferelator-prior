from inferelator_prior.processor.gtf import GTF_GENENAME, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from inferelator_prior.motifs.motif_scan import MotifScan
from inferelator_prior.motifs import INFO_COL, MOTIF_COL, LEN_COL, SCAN_SCORE_COL, MOTIF_NAME_COL, SCORE_PER_BASE

import pandas as pd
import numpy as np
import pathos.multiprocessing as multiprocessing
from sklearn.cluster import DBSCAN

PRIOR_TF = 'regulator'
PRIOR_GENE = 'target'
PRIOR_COUNT = 'count'
PRIOR_SCORE = 'score'
PRIOR_MOTIF_IC = 'motif_ic'
PRIOR_PVAL = 'pvalue'
PRIOR_SEQ = 'sequence'
PRIOR_START = 'start'
PRIOR_STOP = 'stop'
PRIOR_CHR = 'chromosome'

PRIOR_COLS = [PRIOR_TF, PRIOR_GENE, PRIOR_COUNT, PRIOR_SCORE, PRIOR_MOTIF_IC, PRIOR_START, PRIOR_STOP, PRIOR_CHR]

PRIOR_FDR = 'qvalue'
PRIOR_SIG = 'significance'

MINIMUM_MOTIF_IC_BITS = None
MAXIMUM_TANDEM_DISTANCE = 100


class MotifScorer:
    min_binding_ic = MINIMUM_MOTIF_IC_BITS
    max_dist = MAXIMUM_TANDEM_DISTANCE

    @classmethod
    def set_information_criteria(cls, min_binding_ic=None, max_dist=None):
        """
        Set parameters for
        :param min_binding_ic:
        :param max_dist:
        :return:
        """
        cls.min_binding_ic = cls.min_binding_ic if min_binding_ic is None else min_binding_ic
        cls.max_dist = cls.max_dist if max_dist is None else max_dist

    @classmethod
    def score_tf(cls, tf_motifs):
        """
        Score a single TF
        :param tf_motifs: Motif binding sites from FIMO/HOMER
        :type tf_motifs: pd.DataFrame
        :return: Score if the TF should be kept, None otherwise
        """

        assert isinstance(tf_motifs, pd.DataFrame)

        # Drop sites that don't meet threshold
        if cls.min_binding_ic is not None:
            tf_motifs = tf_motifs.loc[tf_motifs[SCAN_SCORE_COL] >= cls.min_binding_ic, :]
        n_sites = tf_motifs.shape[0]

        # If there's no data return None
        if n_sites == 0:
            return None

        # Sort and check for overlapping motifs
        tf_motifs = tf_motifs.sort_values(by=MotifScan.start_col)
        overlap = tf_motifs[MotifScan.start_col] < tf_motifs[MotifScan.stop_col].shift()

        # Collapse together any overlapping motifs to the maximum score on a per-base basis
        if overlap.any():
            tf_motifs["GROUP"] = (~overlap).cumsum()
            tf_motifs = pd.concat([cls._agg_per_base(group) for _, group in tf_motifs.groupby("GROUP")])

            n_sites = tf_motifs.shape[0]

        # If there's only one site check it and then return
        if n_sites == 1:
            return cls._top_hit(tf_motifs)

        # If there's only two sites check it and then return
        if n_sites == 2:
            consider_tandem = tf_motifs.iloc[0, :][MotifScan.stop_col] - tf_motifs.iloc[1, :][MotifScan.start_col]
            if consider_tandem > cls.max_dist:
                return cls._top_hit(tf_motifs)
            else:
                start = tf_motifs.iloc[0, :][MotifScan.start_col]
                stop = tf_motifs.iloc[1, :][MotifScan.stop_col]
                score = tf_motifs[SCAN_SCORE_COL].sum()
                return score, 2, start, stop

        # If there's more than two sites do the complicated tandem checking stuff
        else:
            # Find things that are in tandems
            consider_tandem = (tf_motifs[MotifScan.start_col] - tf_motifs[MotifScan.stop_col].shift(1))
            consider_tandem = consider_tandem <= cls.max_dist

            # Skip the rest if nothing is close enough to matter
            if not consider_tandem.any():
                return cls._top_hit(tf_motifs)

            # Ffill the tandem group to have the same start
            tandem_starts = tf_motifs[MotifScan.start_col].copy()
            tandem_starts.loc[consider_tandem] = pd.NA
            tandem_starts = tandem_starts.ffill()

            # Backfill the tandem group to have the same stop
            tandem_stops = tf_motifs[MotifScan.stop_col].copy()
            tandem_stops.loc[consider_tandem.shift(-1, fill_value=False)] = pd.NA
            tandem_stops = tandem_stops.bfill()

            # Concat, group by start/stop, and then sum IC scores
            tandem_peaks = pd.concat([tandem_starts, tandem_stops, tf_motifs[SCAN_SCORE_COL]], axis=1)
            tandem_peaks.columns = [PRIOR_START, PRIOR_STOP, PRIOR_SCORE]
            tandem_peaks = tandem_peaks.groupby(by=[PRIOR_START, PRIOR_STOP]).agg('sum').reset_index()

            # Return the highest tandem array group
            peak = tandem_peaks.loc[tandem_peaks[PRIOR_SCORE].argmax(), :]
            return peak[PRIOR_SCORE], peak.shape[0], peak[PRIOR_START], peak[PRIOR_STOP]

    @classmethod
    def preprocess_motifs(cls, gene_motif_data, motif_information):
        if cls.min_binding_ic is not None:
            motif_information = motif_information.loc[motif_information[INFO_COL] >= cls.min_binding_ic, :]
            keeper_motifs = motif_information[MOTIF_COL].unique().tolist()
            keeper_idx = (gene_motif_data[MotifScan.name_col].isin(keeper_motifs))
            keeper_idx &= (gene_motif_data[SCAN_SCORE_COL] >= cls.min_binding_ic)

            return gene_motif_data.loc[keeper_idx, :], motif_information
        else:
            return gene_motif_data, motif_information

    @staticmethod
    def _top_hit(tf_motifs):
        if tf_motifs.shape[0] == 0:
            return None
        elif tf_motifs.shape[0] == 1:
            top_hit = tf_motifs.iloc[0, :]
        else:
            top_hit = tf_motifs.iloc[tf_motifs[SCAN_SCORE_COL].values.argmax(), :]

        start = MotifScorer._first_value(top_hit[MotifScan.start_col])
        stop = MotifScorer._first_value(top_hit[MotifScan.stop_col])
        score = MotifScorer._first_value(top_hit[SCAN_SCORE_COL])
        return score, 1, start, stop

    @staticmethod
    def _first_value(series):
        try:
            return series.iloc[0]
        except AttributeError:
            return series

    @staticmethod
    def _agg_per_base(overlap_df):
        """
        Aggregate an overlapping set of motif peaks by summing the maximum per-base IC for each base
        :param overlap_df:
        :return:
        """
        if len(overlap_df) == 1:
            return overlap_df[[MotifScan.start_col, MotifScan.stop_col, SCAN_SCORE_COL, MOTIF_NAME_COL]]

        overlap_df.reset_index(inplace=True)

        # Melt the per-base information contents for each matching motif into a new dataframe
        # Base number ["B"] and float score ["S"]
        new_df = pd.DataFrame([(a, b) for i in overlap_df.index
                               for a, b in zip(range(overlap_df.loc[i, MotifScan.start_col],
                                                     overlap_df.loc[i, MotifScan.stop_col] + 1),
                                               overlap_df.loc[i, SCORE_PER_BASE])], columns=["B", "S"])

        # Return a new dataframe with the maximum per-base scores aggregated
        return pd.DataFrame({MotifScan.start_col: [overlap_df[MotifScan.start_col].min()],
                             MotifScan.stop_col: [overlap_df[MotifScan.stop_col].max()],
                             SCAN_SCORE_COL: new_df.groupby("B").agg('max').sum(),
                             MOTIF_NAME_COL: [overlap_df[MOTIF_NAME_COL].unique()[0]]})


def summarize_target_per_regulator(genes, motif_peaks, motif_information, num_workers=None, debug=False,
                                   by_chromosome=True, silent=False):
    """
    Process a large dataframe of motif hits into a dataframe with the best hit for each regulator-target pair
    :param genes: pd.DataFrame [G x n]
    :param motif_peaks: pd.DataFrame
        Motif search data loaded from FIMO or HOMER
    :param motif_information: pd.DataFrame [n x 5]
        Motif characteristics loaded from a MEME file
    :param num_workers: int
        Number of cores to use
    :return summarized_data: pd.DataFrame [G x K]
        A information matrix connecting genes and regulators
    """

    pfunc = print if not silent else lambda *x: None

    motif_ids = motif_information[MOTIF_COL].unique()
    motif_names = motif_information[MOTIF_NAME_COL].unique()
    pfunc("Building prior from {g} genes and {k} Motifs ({t} TFs)".format(g=genes.shape[0], k=len(motif_ids),
                                                                          t=len(motif_names)))

    motif_peaks, motif_information = MotifScorer.preprocess_motifs(motif_peaks, motif_information)
    pfunc("Preliminary search identified {n} binding sites".format(n=motif_peaks.shape[0]))

    # Trim down the motif dataframe and put it into a dict by chromosome
    motif_peaks = motif_peaks.reindex([MotifScan.name_col, MotifScan.chromosome_col, MotifScan.start_col,
                                       MotifScan.stop_col, SCAN_SCORE_COL, SCORE_PER_BASE], axis=1)

    motif_id_to_name = motif_information.reindex([MOTIF_COL, MOTIF_NAME_COL], axis=1)
    invalid_names = (pd.isnull(motif_id_to_name[MOTIF_NAME_COL]) |
                     (motif_id_to_name[MOTIF_NAME_COL] == "") |
                     (motif_id_to_name is None))

    motif_id_to_name.loc[invalid_names, MOTIF_NAME_COL] = motif_id_to_name.loc[invalid_names, MOTIF_COL]
    motif_peaks = motif_peaks.join(motif_id_to_name.set_index(MOTIF_COL, verify_integrity=True), on=MotifScan.name_col)

    motif_peaks = {chromosome: df for chromosome, df in motif_peaks.groupby(MotifScan.chromosome_col)}

    _gen_func = _gene_gen if by_chromosome else _gene_gen_no_chromosome

    if num_workers == 1:
        prior_data = list(map(lambda x: _build_prior_for_gene(*x),
                              _gen_func(genes, motif_peaks, motif_information, debug=debug, silent=silent)))

    else:
        with multiprocessing.Pool(num_workers, maxtasksperchild=1000) as pool:
            prior_data = pool.starmap(_build_prior_for_gene,
                                      _gen_func(genes, motif_peaks, motif_information, debug=debug, silent=silent),
                                      chunksize=20)

    # Combine priors for all genes
    prior_data = pd.concat(prior_data).reset_index(drop=True)
    prior_data[PRIOR_START] = prior_data[PRIOR_START].astype(int)
    prior_data[PRIOR_STOP] = prior_data[PRIOR_STOP].astype(int)

    # Pivot to a matrix, extend to all TFs, and fill with 0s
    summarized_data = prior_data.pivot(index=PRIOR_GENE, columns=PRIOR_TF, values=PRIOR_SCORE)
    summarized_data = summarized_data.reindex(motif_names, axis=1).reindex(genes[GTF_GENENAME], axis=0).fillna(0)
    summarized_data.index.name = PRIOR_GENE

    return summarized_data, prior_data


def build_prior_from_motifs(raw_matrix, num_workers=None, seed=42, do_threshold=True, debug=False, silent=False):
    """
    Construct a prior [G x K] interaction matrix
    :param raw_matrix: pd.DataFrame [G x K]
        Scored matrix between targets and regulators
    :param num_workers: int
        Number of cores to use
    :param seed: int
        Random seed for numpy random pool
    :param do_threshold: bool
        Threshold using DBSCAN if true; retain all non-zero edges if false
    :return prior_matrix: pd.DataFrame [G x K]
        An interaction matrix data frame
    """

    np.random.seed(seed)
    pfunc = print if not silent else lambda *x: None

    if do_threshold:
        # Threshold per-TF using DBSCAN
        pfunc("Selecting edges to retain with DBSCAN")
        prior_matrix = pd.DataFrame(False, index=raw_matrix.index, columns=raw_matrix.columns)

        if num_workers == 1:
            prior_matrix_idx = list(map(lambda x: _prior_clusterer(*x),
                                        _prior_gen(raw_matrix, debug=debug, silent=silent)))

        else:
            with multiprocessing.Pool(num_workers, maxtasksperchild=1) as pool:
                prior_matrix_idx = pool.starmap(_prior_clusterer, _prior_gen(raw_matrix, debug=debug, silent=silent),
                                                chunksize=1)

        pfunc("Completed edge selection with DBSCAN")
        for reg, reg_idx in prior_matrix_idx:
            prior_matrix.loc[reg_idx, reg] = True

        return prior_matrix

    else:
        pfunc("Retaining all edges")
        return raw_matrix != 0


def _prior_gen(prior_matrix, debug=False, silent=False):

    n = len(prior_matrix.columns)

    for i, col_name in enumerate(prior_matrix.columns):
        yield i, col_name, prior_matrix[col_name], n, debug, silent


def _prior_clusterer(i, col_name, col_data, n, debug=False, silent=False):

    pfunc = print if not silent else lambda *x: None

    if not debug and (i % 50 == 0):
        pfunc("Clustering {col} [{i} / {n}]".format(i=i, n=n, col=col_name))

    keep_idx = _find_outliers_dbscan(col_data)

    if debug:
        pfunc("Keeping {ed} edges for gene {col} [{i} / {n}]".format(ed=keep_idx.sum(), i=i, n=n, col=col_name))

    return col_name, keep_idx


def _gene_gen_no_chromosome(genes, motif_peaks, motif_information, debug=False, silent=False):
    """
    Yield the peaks for each group by seqname (which should be the gene promoter)

    :param genes:
    :param motif_peaks:
    :param motif_information:
    :param debug:
    :return:
    """
    for i, gene in enumerate(motif_peaks.keys()):
        gene_loc = {GTF_GENENAME: gene, GTF_CHROMOSOME: None}

        if i % 100 == 0 and not silent:
            print("Processing gene {i} [{gn}]".format(i=i, gn=gene))

        yield gene_loc, motif_peaks[gene], motif_information


def _gene_gen(genes, motif_peaks, motif_information, debug=False, silent=False):
    """
    Yield the peaks for each gene

    :param genes:
    :param motif_peaks:
    :param motif_information:
    :param debug:
    :return:
    """
    gene_names = genes[GTF_GENENAME].unique().tolist()
    bad_chr = {}

    pfunc = print if not silent else lambda *x: None

    for i, gene in enumerate(gene_names):
        gene_data = genes.loc[genes[GTF_GENENAME] == gene, :]
        gene_loc = {GTF_GENENAME: gene, GTF_CHROMOSOME: gene_data.iloc[0, :][GTF_CHROMOSOME]}

        if i % 100 == 0:
            pfunc("Processing gene {i} [{gn}]".format(i=i, gn=gene))

        gene_motifs = []
        for _, row in gene_data.iterrows():
            gene_chr, gene_start, gene_stop = row[GTF_CHROMOSOME], row[SEQ_START], row[SEQ_STOP]

            try:
                motif_data = motif_peaks[gene_chr]
            except KeyError:
                # If this chromosome is some weird scaffold or not in the genome, skip it
                pfunc("Chromosome {c} not found; skipping gene {g}".format(c=gene_chr, g=gene)) if debug else None

                if gene_chr not in bad_chr.keys():
                    bad_chr[gene_chr] = 1
                else:
                    bad_chr[gene_chr] += 1

                continue

            motif_mask = motif_data[MotifScan.stop_col] >= gene_start
            motif_mask &= motif_data[MotifScan.start_col] <= gene_stop
            gene_motifs.append(motif_data.loc[motif_mask, :])

        if len(gene_motifs) == 0:
            continue

        gene_motifs = pd.concat(gene_motifs)
        yield gene_loc, gene_motifs, motif_information

    for chromosome, bad_genes in bad_chr.items():
        pfunc("{n} genes annotated to chromosome {c} have been skipped".format(n=bad_genes, c=chromosome))


def _find_outliers_dbscan(tf_data, max_sparsity=0.05):
    scores, weights = np.unique(tf_data.values, return_counts=True)

    labels = DBSCAN(min_samples=max(int(scores.size * 0.001), 10), eps=1, n_jobs=None)\
        .fit_predict(scores.reshape(-1, 1), sample_weight=weights)

    # Short circuit if all the labels are outliers
    # This shouldn't happen real-world unless there aren't many genes in the network
    if np.all(labels == -1):
        return pd.Series(tf_data.values > 0, index=tf_data.index)

    # Short circuit if all the labels are in the same cluster
    if np.all(labels == 0):
        return pd.Series(False, index=tf_data.index)

    largest_cluster = np.argmax(np.array([np.min(scores[labels == i]) for i in range(np.max(labels) + 1)]))
    min_score = np.min(scores[labels == largest_cluster])

    # If the largest cluster is less than max_sparsity, keep it and any outliers greater than it
    keep_all_values = tf_data >= min_score
    if keep_all_values.sum() / keep_all_values.size <= max_sparsity:
        return keep_all_values

    # If the largest cluster exceeds max_sparsity, only keep outliers
    else:
        keep_outlier_values = scores[(labels == -1) & (scores > min_score)]
        return pd.Series(np.isin(tf_data.values, keep_outlier_values), index=tf_data.index)


def _build_prior_for_gene(gene_info, motif_data, motif_information):
    """
    Takes motifs identified by scan near a single gene and turns them into TF-gene scores

    :param gene_info: Gene information from annotations
    :type gene_info: pd.DataFrame
    :param motif_data: Motif locations near the gene
    :type motif_data: pd.DataFrame
    :param motif_information: Motif information
    :type motif_information: pd.DataFrame
    :return prior_edges: pd.DataFrame [N x 5]
        'regulator': tf name
        'target': gene name
        'count': number of motifs found
        'score': information content-based score of binding site
        'motif_ic': information content score of motif
        'start': binding site start
        'stop': binding site stop
        'chromosome' binding site chromosome
    """

    gene_name, gene_chr = gene_info[GTF_GENENAME], gene_info[GTF_CHROMOSOME]

    if min(motif_data.shape) == 0:
        return pd.DataFrame(columns=PRIOR_COLS)

    prior_edges = []
    for tf, tf_peaks in motif_data.groupby(MOTIF_NAME_COL):
        tf_info = motif_information.loc[motif_information[MOTIF_NAME_COL] == tf, :]
        res = MotifScorer.score_tf(tf_peaks)

        # Unpack results if there is a hit
        if res is None:
            continue
        else:
            score, tf_counts, start, stop = res

        info = tf_info[INFO_COL].mean() if tf_info.shape[0] > 0 else np.nan

        # Add this edge to the table
        prior_edges.append((tf, gene_name, tf_counts, score, info, start, stop, gene_chr))

    return pd.DataFrame(prior_edges, columns=PRIOR_COLS)
