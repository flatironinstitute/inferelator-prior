from srrTomat0.processor.gtf import GTF_GENENAME, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.motifs.motif_locations import MotifLocationManager as MotifLM
from srrTomat0.motifs import INFO_COL, MOTIF_COL, LEN_COL

import pandas as pd
import numpy as np

PRIOR_TF = 'regulator'
PRIOR_GENE = 'target'
PRIOR_COUNT = 'count'
PRIOR_SCORE = 'score'
PRIOR_PVAL = 'pvalue'

PRIOR_COLS = [PRIOR_TF, PRIOR_GENE, PRIOR_COUNT, PRIOR_SCORE, PRIOR_PVAL]

PRIOR_FDR = 'qvalue'
PRIOR_SIG = 'significance'

MINIMUM_IC_BITS = 8
MINIMUM_TANDEM_ARRAY = 4
TANDEM_STEP_BITS = 4
MAXIMUM_TANDEM_DISTANCE = 200


class MotifScorer:
    min_ic = MINIMUM_IC_BITS
    min_tandem = MINIMUM_TANDEM_ARRAY

    min_single_ic = MINIMUM_IC_BITS + (MINIMUM_TANDEM_ARRAY - 1) * TANDEM_STEP_BITS
    step_ic = TANDEM_STEP_BITS

    max_dist = MAXIMUM_TANDEM_DISTANCE

    @classmethod
    def score_tf(cls, gene_motif_data, motif_ic, motif_len=None):
        """
        Score a single TF
        :param gene_motif_data: Motif binding sites from FIMO/HOMER
        :type gene_motif_data: pd.DataFrame
        :param motif_ic: Information content (bits)
        :type motif_ic: float
        :param motif_len: Length of the motif recognition site
        :type motif_len: int
        :return: Score if the TF should be kept, None otherwise
        """

        assert isinstance(motif_ic, float)
        assert isinstance(gene_motif_data, pd.DataFrame)
        assert motif_len is None or isinstance(motif_len, int)

        n_sites = gene_motif_data.shape[0]

        # If there's no data return None
        if n_sites == 0:
            return None

        # Discard weak motifs
        if motif_ic < cls.min_ic or min(0, cls.step_ic * (cls.min_tandem - n_sites)) + cls.min_ic > motif_ic:
            return None

        # Return a score if the motif is strong enough to not need tandem array
        if motif_ic > cls.min_single_ic:
            return cls._score(n_sites, motif_ic)

        # Calculate the required number of tandems for this motif
        req_tandem = int((cls.min_single_ic - motif_ic) / cls.step_ic)

        # Skip if there's too few sites to possibly pass
        if req_tandem > n_sites:
            return None

        starts = gene_motif_data[MotifLM.start_col].sort_values()
        consider_tandem = (starts - starts.shift(1)) <= cls.max_dist if motif_len is None else cls.max_dist + motif_len

        if n_sites == 2 and consider_tandem < cls.max_dist if motif_len is None else cls.max_dist + motif_len:
            return cls._score(1, (motif_ic + cls.step_ic * req_tandem))
        elif n_sites == 2:
            return None

        # Calculate the actual number of tandems in this
        ct_cumsum = consider_tandem.cumsum()
        ct_cumsum = ct_cumsum.sub(ct_cumsum.mask(consider_tandem).ffill().fillna(0)).astype(int)
        passed_tandems = (ct_cumsum >= req_tandem).sum()

        if passed_tandems > 0:
            return cls._score(passed_tandems, (motif_ic + cls.step_ic * req_tandem))
        else:
            return None

    @classmethod
    def preprocess_motifs(cls, gene_motif_data, motif_information):
        motif_information = motif_information.loc[motif_information[INFO_COL] < cls.min_ic, :]
        keeper_motifs = motif_information[MOTIF_COL].unique()
        gene_motif_data = gene_motif_data.loc[gene_motif_data[MotifLM.name_col].str.contains(keeper_motifs), :]
        return gene_motif_data, motif_information

    @staticmethod
    def _score(n_sites, motif_ic):
        return n_sites * motif_ic * np.log10(2)


def build_prior_from_atac_motifs(genes, motif_peaks, motif_information):
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

    motif_names = motif_information[MOTIF_COL].unique()
    print("Building prior from {g} genes and {k} TFs".format(g=genes.shape[0], k=len(motif_names)))

    # Trim down the motif dataframe and put it into a dict by chromosome
    motif_peaks = motif_peaks.reindex([MotifLM.name_col, MotifLM.chromosome_col, MotifLM.start_col, MotifLM.stop_col],
                                      axis=1)
    motif_peaks = {chromosome: df for chromosome, df in motif_peaks.groupby(MotifLM.chromosome_col)}

    def _prior_mapper(data):
        i, gene_data, motifs = data
        return _build_prior_for_gene(gene_data, motifs, motif_information, i)

    prior_data = list(map(_prior_mapper, _gene_gen(genes, motif_peaks)))

    # Combine priors for all genes
    prior_data = pd.concat(prior_data)

    # Pivot to a matrix, extend to all TFs, and fill with 1s
    prior_matrix = prior_data.pivot(index=PRIOR_GENE, columns=PRIOR_TF, values=PRIOR_SCORE)
    prior_matrix = prior_matrix.reindex(motif_names, axis=1)
    prior_matrix = prior_matrix.reindex(genes[GTF_GENENAME], axis=0)
    prior_matrix[pd.isnull(prior_matrix)] = 0

    return prior_data, prior_matrix


def _gene_gen(genes, motif_peaks):
    for i, (idx, gene_data) in enumerate(genes.iterrows()):
        try:
            yield i, gene_data, motif_peaks[gene_data[GTF_CHROMOSOME]]
        except KeyError:
            continue


def _build_prior_for_gene(gene_info, motif_peaks, motif_information, num_iteration):
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

    motif_mask = motif_peaks[MotifLM.chromosome_col] == gene_chr
    motif_mask &= motif_peaks[MotifLM.stop_col] >= gene_start
    motif_mask &= motif_peaks[MotifLM.start_col] <= gene_stop
    motif_data = motif_peaks.loc[motif_mask, :]

    if num_iteration % 100 == 0:
        print("Processing gene {i} [{gn}]".format(i=num_iteration, gn=gene_name))

    if min(motif_data.shape) == 0:
        return pd.DataFrame(columns=PRIOR_COLS)

    prior_edges = []
    for tf, tf_peaks in motif_data.groupby(MotifLM.name_col):
        tf_info = motif_information.loc[motif_information[MOTIF_COL] == tf, :]
        try:
            score = MotifScorer.score_tf(tf_peaks, tf_info[INFO_COL][0], tf_info[LEN_COL][0])
        except KeyError:
            continue

        if score is None:
            continue

        tf_counts = tf_peaks.shape[0]

        # Add this edge to the table
        prior_edges.append((tf, gene_name, tf_counts, score, 10 ** (-1 * score)))

    return pd.DataFrame(prior_edges, columns=PRIOR_COLS)
