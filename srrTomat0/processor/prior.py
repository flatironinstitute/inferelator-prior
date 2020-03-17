from srrTomat0.processor.gtf import GTF_GENENAME, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.motifs.motif_scan import MotifScan
from srrTomat0.motifs import INFO_COL, MOTIF_COL, LEN_COL, SCAN_SCORE_COL, MOTIF_NAME_COL

import pandas as pd
import numpy as np
import pathos.multiprocessing as multiprocessing

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

MINIMUM_MOTIF_IC_BITS = 6
MINIMUM_HIT_BITS = 24
MAXIMUM_TANDEM_DISTANCE = 100


class MotifScorer:
    min_binding_ic = MINIMUM_MOTIF_IC_BITS
    min_hit = MINIMUM_HIT_BITS
    max_dist = MAXIMUM_TANDEM_DISTANCE

    @classmethod
    def set_information_criteria(cls, min_binding_ic=None, min_hit_ic=None, max_dist=None):
        """
        Set parameters for
        :param min_binding_ic:
        :param min_hit_ic:
        :param max_dist:
        :return:
        """
        cls.min_binding_ic = cls.min_binding_ic if min_binding_ic is None else min_binding_ic
        cls.max_dist = cls.max_dist if max_dist is None else max_dist
        cls.min_hit = cls.min_hit if min_hit_ic is None else min_hit_ic

    @classmethod
    def score_tf(cls, gene_motif_data, motif_len=None):
        """
        Score a single TF
        :param gene_motif_data: Motif binding sites from FIMO/HOMER
        :type gene_motif_data: pd.DataFrame
        :param motif_len: Length of the motif recognition site
        :type motif_len: int
        :return: Score if the TF should be kept, None otherwise
        """

        assert isinstance(gene_motif_data, pd.DataFrame)
        assert motif_len is None or isinstance(motif_len, int)

        # Drop sites that don't meet threshold
        gene_motif_data = gene_motif_data.loc[gene_motif_data[SCAN_SCORE_COL] >= cls.min_binding_ic, :]
        n_sites = gene_motif_data.shape[0]

        # If there's no data return None
        if n_sites == 0:
            return None

        # If there's only one site check it and then return
        elif n_sites == 1:
            score = gene_motif_data[SCAN_SCORE_COL].iloc[0]
            if score >= cls.min_hit:
                start, stop = gene_motif_data[MotifScan.start_col].iloc[0], gene_motif_data[MotifScan.stop_col].iloc[0]
                return score, n_sites, start, stop
            else:
                return None

        # If there's more than one site do the tandem checking stuff
        else:
            m_dist = cls.max_dist if motif_len is None else cls.max_dist + motif_len

            gene_motif_data = gene_motif_data.sort_values(by=MotifScan.start_col)

            # Find things that are in tandems
            consider_tandem = (gene_motif_data[MotifScan.start_col] - gene_motif_data[MotifScan.start_col].shift(1))
            consider_tandem = consider_tandem <= m_dist

            # Ffill the tandem group to have the same start
            tandem_starts = gene_motif_data[MotifScan.start_col].copy()
            tandem_starts.loc[consider_tandem] = pd.NA
            tandem_starts = tandem_starts.ffill()

            # Backfill the tandem group to have the same stop
            tandem_stops = gene_motif_data[MotifScan.stop_col].copy()
            tandem_stops.loc[consider_tandem.shift(-1, fill_value=False)] = pd.NA
            tandem_stops = tandem_stops.bfill()

            # Concat, group by start/stop, and then sum IC scores
            tandem_peaks = pd.concat([tandem_starts, tandem_stops, gene_motif_data[SCAN_SCORE_COL]], axis=1)
            tandem_peaks.columns = [PRIOR_START, PRIOR_STOP, PRIOR_SCORE]
            tandem_peaks = tandem_peaks.groupby(by=[PRIOR_START, PRIOR_STOP]).agg('sum').reset_index()

            # If the sum is greater than the IC threshold for a hit then return the tandem range
            if tandem_peaks[PRIOR_SCORE].max() >= cls.min_hit:
                peak = tandem_peaks.loc[tandem_peaks[PRIOR_SCORE].argmax(), :]
                return peak[PRIOR_SCORE], peak.shape[0], peak[PRIOR_START], peak[PRIOR_STOP]
            else:
                return None

    @classmethod
    def preprocess_motifs(cls, gene_motif_data, motif_information):
        motif_information = motif_information.loc[motif_information[INFO_COL] >= cls.min_binding_ic, :]
        keeper_motifs = motif_information[MOTIF_COL].unique().tolist()
        keeper_idx = (gene_motif_data[MotifScan.name_col].isin(keeper_motifs))
        keeper_idx &= (gene_motif_data[SCAN_SCORE_COL] >= cls.min_binding_ic)

        return gene_motif_data.loc[keeper_idx, :], motif_information

    @staticmethod
    def _score(n_sites, motif_ic):
        return n_sites * motif_ic * np.log10(2)


def build_prior_from_atac_motifs(genes, motif_peaks, motif_information, num_workers=1):
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

    # Pivot to a matrix, extend to all TFs, and fill with 1s
    prior_matrix = prior_data.pivot(index=PRIOR_GENE, columns=PRIOR_TF, values=PRIOR_SCORE)
    prior_matrix = prior_matrix.reindex(motif_names, axis=1)
    prior_matrix = prior_matrix.reindex(genes[GTF_GENENAME], axis=0)
    prior_matrix[pd.isnull(prior_matrix)] = 0

    return prior_data, prior_matrix


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
        res = MotifScorer.score_tf(tf_peaks, tf_info[LEN_COL].iloc[0])

        # Unpack results if there is a hit
        if res is None:
            continue
        else:
            score, tf_counts, start, stop = res

        # Add this edge to the table
        prior_edges.append((tf, gene_name, tf_counts, score, tf_info[INFO_COL].mean(), start, stop, gene_chr))

    return pd.DataFrame(prior_edges, columns=PRIOR_COLS)
