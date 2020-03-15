from srrTomat0.processor.gtf import GTF_GENENAME, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.motifs.motif_locations import MotifLocationManager as MotifLM
from srrTomat0.motifs import INFO_COL, MOTIF_COL, LEN_COL, SCAN_SCORE_COL, MOTIF_NAME_COL

import pandas as pd
import numpy as np

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

MINIMUM_IC_BITS = 6
MINIMUM_TANDEM_ARRAY = 3
TANDEM_STEP_BITS = 6
MAXIMUM_TANDEM_DISTANCE = 50


class MotifScorer:
    min_ic = MINIMUM_IC_BITS
    min_tandem = MINIMUM_TANDEM_ARRAY

    min_single_ic = MINIMUM_IC_BITS + (MINIMUM_TANDEM_ARRAY - 1) * TANDEM_STEP_BITS
    step_ic = TANDEM_STEP_BITS

    max_dist = MAXIMUM_TANDEM_DISTANCE

    @classmethod
    def set_information_criteria(cls, min_ic=None, min_tandem=None, step_ic=None, max_dist=None):
        """
        Set parameters for
        :param min_ic:
        :param min_tandem:
        :param step_ic:
        :param max_dist:
        :return:
        """
        cls.min_ic = cls.min_ic if min_ic is None else min_ic
        cls.min_tandem = cls.min_tandem if min_tandem is None else min_tandem
        cls.step_ic = cls.step_ic if step_ic is None else step_ic
        cls.max_dist = cls.max_dist if max_dist is None else max_dist
        cls.min_single_ic = cls.min_ic + (cls.min_tandem - 1) * cls.step_ic

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
        gene_motif_data = gene_motif_data.loc[gene_motif_data[SCAN_SCORE_COL] >= cls.min_ic, :]
        n_sites = gene_motif_data.shape[0]

        # If there's no data return None
        if n_sites == 0:
            return None
        elif n_sites == 1:
            score = gene_motif_data[SCAN_SCORE_COL].iloc[0]
            if score >= cls.min_single_ic:
                return score, cls._peak_region(gene_motif_data)
            else:
                return None

        m_dist = cls.max_dist if motif_len is None else cls.max_dist + motif_len

        gene_motif_data = gene_motif_data.sort_values(by=MotifLM.start_col)

        # Add stepwise boost to information from tandems
        consider_tandem = (gene_motif_data[MotifLM.start_col] - gene_motif_data[MotifLM.start_col].shift(1)) <= m_dist
        ct_cumsum = consider_tandem.cumsum()
        ct_cumsum = ct_cumsum.sub(ct_cumsum.mask(consider_tandem).ffill().fillna(0)).astype(int)

        gene_motif_data[SCAN_SCORE_COL] = gene_motif_data[SCAN_SCORE_COL] + ct_cumsum * cls.step_ic
        max_score = gene_motif_data[SCAN_SCORE_COL].max()

        if max_score < cls.min_single_ic:
            return None

        stop_line = gene_motif_data[SCAN_SCORE_COL].argmax()
        start_line = stop_line - ct_cumsum.iloc[stop_line]

        if start_line != stop_line:
            return max_score, cls._peak_region(gene_motif_data.iloc[start_line:stop_line, :])
        else:
            return max_score, cls._peak_region(gene_motif_data.iloc[start_line, :])

    @classmethod
    def preprocess_motifs(cls, gene_motif_data, motif_information):
        motif_information = motif_information.loc[motif_information[INFO_COL] >= cls.min_ic, :]
        keeper_motifs = motif_information[MOTIF_COL].unique().tolist()
        keeper_idx = (gene_motif_data[MotifLM.name_col].isin(keeper_motifs))
        keeper_idx &= (gene_motif_data[SCAN_SCORE_COL] >= cls.min_ic)

        return gene_motif_data.loc[keeper_idx, :], motif_information

    @staticmethod
    def _score(n_sites, motif_ic):
        return n_sites * motif_ic * np.log10(2)

    @staticmethod
    def _peak_region(peaks):
        start = peaks[MotifLM.start_col].min()
        stop = peaks[MotifLM.stop_col].max()
        chromosome = peaks[MotifLM.chromosome_col]
        chromosome = chromosome.iloc[0] if isinstance(chromosome, pd.Series) else chromosome
        return start, stop, chromosome


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

    motif_ids = motif_information[MOTIF_COL].unique()
    print("Building prior from {g} genes and {k} Motifs".format(g=genes.shape[0], k=len(motif_ids)))

    motif_peaks, motif_information = MotifScorer.preprocess_motifs(motif_peaks, motif_information)
    print("Preliminary search identified {n} binding sites".format(n=motif_peaks.shape[0]))

    # Trim down the motif dataframe and put it into a dict by chromosome
    motif_peaks = motif_peaks.reindex([MotifLM.name_col, MotifLM.chromosome_col, MotifLM.start_col, MotifLM.stop_col,
                                       SCAN_SCORE_COL], axis=1)

    motif_id_to_name = motif_information.reindex([MOTIF_COL, MOTIF_NAME_COL], axis=1)
    invalid_names = (pd.isnull(motif_id_to_name[MOTIF_NAME_COL]) |
                     (motif_id_to_name[MOTIF_NAME_COL] == "") |
                     (motif_id_to_name is None))

    motif_id_to_name.loc[invalid_names, MOTIF_NAME_COL] = motif_id_to_name.loc[invalid_names, MOTIF_COL]
    motif_peaks = motif_peaks.join(motif_id_to_name.set_index(MOTIF_COL, verify_integrity=True), on=MotifLM.name_col)
    motif_names = motif_information[MOTIF_NAME_COL].unique()

    motif_peaks = {chromosome: df for chromosome, df in motif_peaks.groupby(MotifLM.chromosome_col)}

    def _prior_mapper(data):
        i, gene_data, motifs = data
        return _build_prior_for_gene(gene_data, motifs, motif_information, i)

    prior_data = list(map(_prior_mapper, _gene_gen(genes, motif_peaks)))

    # Combine priors for all genes
    prior_data = pd.concat(prior_data).reset_index(drop=True)

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

    motif_mask = motif_peaks[MotifLM.stop_col] >= gene_start
    motif_mask &= motif_peaks[MotifLM.start_col] <= gene_stop
    motif_data = motif_peaks.loc[motif_mask, :]

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
            score, (start, stop, chromosome) = res

        tf_counts = tf_peaks.shape[0]

        # Add this edge to the table
        prior_edges.append((tf, gene_name, tf_counts, score, tf_info[INFO_COL].iloc[0], start, stop, chromosome))

    return pd.DataFrame(prior_edges, columns=PRIOR_COLS)
