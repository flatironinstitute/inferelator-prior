from srrTomat0.processor.gtf import GTF_GENENAME, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.motifs.motif_scan import MotifScan
from srrTomat0.motifs import INFO_COL, MOTIF_COL, LEN_COL, SCAN_SCORE_COL, MOTIF_NAME_COL

import pandas as pd
import numpy as np
import pathos.multiprocessing as multiprocessing
from sklearn.cluster import DBSCAN
from collections import Counter


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
        :param motif_len: Length of the motif recognition site
        :type motif_len: int
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

        # If there's only one site check it and then return
        if n_sites == 1:
            return cls._top_hit(tf_motifs)

        tf_motifs = tf_motifs.sort_values(by=MotifScan.start_col)

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
            consider_tandem = (tf_motifs[MotifScan.stop_col] - tf_motifs[MotifScan.start_col].shift(1))
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


def build_prior_from_atac_motifs(genes, motif_peaks, motif_information, num_workers=1, seed=42):
    """
    Construct a prior [G x K] interaction matrix
    :param genes: pd.DataFrame [G x n]
    :param motif_peaks: pd.DataFrame
        Motif search data loaded from FIMO or HOMER
    :param motif_information: pd.DataFrame [n x 5]
        Motif characteristics loaded from a MEME file
    :return prior_data, prior_matrix: pd.DataFrame [G*K x 6], pd.DataFrame [G x K]
        A long-form edge table data frame and a wide-form interaction matrix data frame
    """

    motif_ids = motif_information[MOTIF_COL].unique()
    print("Building prior from {g} genes and {k} Motifs".format(g=genes.shape[0], k=len(motif_ids)))

    motif_peaks, motif_information = MotifScorer.preprocess_motifs(motif_peaks, motif_information)
    print("Preliminary search identified {n} binding sites".format(n=motif_peaks.shape[0]))

    # Trim down the motif dataframe and put it into a dict by chromosome
    motif_peaks = motif_peaks.reindex([MotifScan.name_col, MotifScan.chromosome_col, MotifScan.start_col,
                                       MotifScan.stop_col, SCAN_SCORE_COL], axis=1)

    motif_id_to_name = motif_information.reindex([MOTIF_COL, MOTIF_NAME_COL], axis=1)
    invalid_names = (pd.isnull(motif_id_to_name[MOTIF_NAME_COL]) |
                     (motif_id_to_name[MOTIF_NAME_COL] == "") |
                     (motif_id_to_name is None))

    motif_id_to_name.loc[invalid_names, MOTIF_NAME_COL] = motif_id_to_name.loc[invalid_names, MOTIF_COL]
    motif_peaks = motif_peaks.join(motif_id_to_name.set_index(MOTIF_COL, verify_integrity=True), on=MotifScan.name_col)
    motif_names = motif_information[MOTIF_NAME_COL].unique()

    motif_peaks = {chromosome: df for chromosome, df in motif_peaks.groupby(MotifScan.chromosome_col)}

    def _prior_mapper(data):
        i, gene_data, motifs = data
        return _build_prior_for_gene(gene_data, motifs, motif_information, i)

    if num_workers == 1:
        prior_data = list(map(_prior_mapper, _gene_gen(genes, motif_peaks)))
    else:
        with multiprocessing.Pool(num_workers, maxtasksperchild=1000) as pool:
            prior_data = pool.map(_prior_mapper, _gene_gen(genes, motif_peaks), chunksize=20)

    # Combine priors for all genes
    prior_data = pd.concat(prior_data).reset_index(drop=True)
    prior_data[PRIOR_START] = prior_data[PRIOR_START].astype(int)
    prior_data[PRIOR_STOP] = prior_data[PRIOR_STOP].astype(int)

    np.random.seed(seed)

    target_size = int(0.005 * genes.shape[0])
    thresholded_data = []
    # Threshold using DBSCAN outlier detection
    for reg in prior_data[PRIOR_TF].unique():
        reg_edge = prior_data.loc[prior_data[PRIOR_TF] == reg, :]
        if reg_edge.shape[0] > target_size:
            reg_edge = reg_edge.loc[_find_outliers_dbscan(reg_edge), :]
        thresholded_data.append(reg_edge.copy())

    thresholded_data = pd.concat(thresholded_data).reset_index(drop=True)

    # Pivot to a matrix, extend to all TFs, and fill with 1s
    prior_matrix = thresholded_data.pivot(index=PRIOR_GENE, columns=PRIOR_TF, values=PRIOR_SCORE)
    prior_matrix = prior_matrix.reindex(motif_names, axis=1).reindex(genes[GTF_GENENAME], axis=0).fillna(0)

    # Pivot to a matrix, extend to all TFs, and fill with 1s
    raw_matrix = prior_data.pivot(index=PRIOR_GENE, columns=PRIOR_TF, values=PRIOR_SCORE)
    raw_matrix = raw_matrix.reindex(motif_names, axis=1).reindex(genes[GTF_GENENAME], axis=0).fillna(0)

    return thresholded_data, prior_matrix, raw_matrix


def _gene_gen(genes, motif_peaks):
    for i, (idx, gene_data) in enumerate(genes.iterrows()):
        try:
            gene_chr, gene_start, gene_stop = gene_data[GTF_CHROMOSOME], gene_data[SEQ_START], gene_data[SEQ_STOP]

            motif_data = motif_peaks[gene_data[GTF_CHROMOSOME]]
            motif_mask = motif_data[MotifScan.stop_col] >= gene_start
            motif_mask &= motif_data[MotifScan.start_col] <= gene_stop
            motif_data = motif_data.loc[motif_mask, :].copy()
            yield i, gene_data, motif_data
        except KeyError:
            continue


def _find_outliers_dbscan(tf_data):
    scores = tf_data[PRIOR_SCORE].values.reshape(-1, 1)
    counts = tf_data.shape[0]

    labels = DBSCAN(min_samples=np.sqrt(counts), eps=1).fit_predict(scores)
    outlier_labels = labels == -1

    mean_score = np.mean(scores)
    keep_edge = pd.Series(outlier_labels & (tf_data[PRIOR_SCORE].values > mean_score), index=tf_data.index)

    # Check the highest non-outlier cluster to see if it's worth including
    lbl_idx = labels == (labels[scores[~keep_edge.values].argmax()])
    if (np.min(scores[lbl_idx]) > mean_score) and (np.sum(lbl_idx) < (2 * np.sum(outlier_labels))):
        keep_edge |= lbl_idx

    return keep_edge


def _build_prior_for_gene(gene_info, motif_data, motif_information, num_iteration):
    """
    Takes ATAC peaks and Motif locations near a single gene and turns them into TF-gene scores

    :param gene_data: (str, pd.DataFrame, int, pd.DataFrame)
        Unpacks to gene_name, motif_data, num_iteration, motif_data
        gene_name: str identifier for the gene
        chromatin_data: pd.DataFrame which has the ATAC (open chromatin) peaks near the gene
        motif_data: pd.DataFrame which has the Motif locations near the gene
        num_iteration: int the number of genes which have been processed
    :return prior_edges: pd.DataFrame [N x 5]
        'regulator': tf name
        'target': gene name
        'count': number of motifs found
        'score': negative log10 of p-value
        'pvalue': p-value calculated using poisson survival function
    """

    gene_name = gene_info[GTF_GENENAME]
    gene_chr, gene_start, gene_stop = gene_info[GTF_CHROMOSOME], gene_info[SEQ_START], gene_info[SEQ_STOP]

    if num_iteration % 100 == 0:
        print("Processing gene {i} [{gn}]".format(i=num_iteration, gn=gene_name))

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
