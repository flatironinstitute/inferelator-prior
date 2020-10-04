import warnings
import pybedtools
import pandas as pd

GENE_ID_REGEX = 'gene_id\s\"([A-Za-z0-9\.\-\(\)]+)\"\;'

# Column names
GTF_ATTRIBUTES = 'attributes'
GTF_CHROMOSOME = 'seqname'
GTF_GENENAME = 'gene_name'
GTF_STRAND = 'strand'
SEQ_START = 'start'
SEQ_STOP = 'end'
SEQ_TSS = 'TSS'


def load_gtf_to_dataframe(gtf_path):
    """
    Loads genes from a GTF into a dataframe and returns them
    :param gtf_path: str
    :return annotations: pd.DataFrame [N x 5]
        'gene_name': str
        'strand': str
        'start': int
        'end': int
        'seqname': str
    """

    # Load annotations into a dataframe with pybedtools
    annotations = pybedtools.BedTool(gtf_path).to_dataframe()

    # Drop anything with NaNs which were probably comment lines
    annotations = annotations.loc[~pd.isnull(annotations[SEQ_START]) & ~pd.isnull(annotations[SEQ_STOP]), :]

    # Regex extract the gene_id from the annotations column
    annotations[GTF_GENENAME] = annotations[GTF_ATTRIBUTES].str.extract(GENE_ID_REGEX, expand=False)

    # Define genes as going from the minimum start for any subfeature to the maximum end for any subfeature
    annotations = _fix_genes(annotations)

    # Fix chromosome names to always be strings
    annotations[GTF_CHROMOSOME] = annotations[GTF_CHROMOSOME].astype(str)

    return _add_TSS(annotations)


def open_window(annotation_dataframe, window_size, use_tss=False, check_against_fasta=None):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :param use_tss: bool
    :param check_against_fasta:
    :return window_annotate: pd.DataFrame
    """
    window_annotate = annotation_dataframe.copy()

    try:
        if len(window_size) == 1:
            w_up, w_down = window_size[0], window_size[0]
        elif len(window_size) == 2:
            w_up, w_down = window_size[0], window_size[1]
        else:
            raise ValueError("window_size must have 1 or 2 values only")
    except TypeError:
        w_up, w_down = window_size, window_size

    if use_tss:
        window_annotate.loc[window_annotate[GTF_STRAND] == "+", SEQ_START] = window_annotate[SEQ_TSS] - w_up
        window_annotate.loc[window_annotate[GTF_STRAND] == "+", SEQ_STOP] = window_annotate[SEQ_TSS] + w_down
        window_annotate.loc[window_annotate[GTF_STRAND] == "-", SEQ_START] = window_annotate[SEQ_TSS] - w_down
        window_annotate.loc[window_annotate[GTF_STRAND] == "-", SEQ_STOP] = window_annotate[SEQ_TSS] + w_up
    else:
        window_annotate.loc[window_annotate[GTF_STRAND] == "+", SEQ_START] = window_annotate[SEQ_START] - w_up
        window_annotate.loc[window_annotate[GTF_STRAND] == "+", SEQ_STOP] = window_annotate[SEQ_STOP] + w_down
        window_annotate.loc[window_annotate[GTF_STRAND] == "-", SEQ_START] = window_annotate[SEQ_START] - w_down
        window_annotate.loc[window_annotate[GTF_STRAND] == "-", SEQ_STOP] = window_annotate[SEQ_STOP] + w_up

    window_annotate.loc[window_annotate[SEQ_START] < 1, SEQ_START] = 1

    if check_against_fasta is not None:
        fasta_len = {}
        with open(check_against_fasta, mode="r") as fasta_fh:
            current_record = None
            for line in fasta_fh:
                if line.startswith(">"):
                    current_record = line[1:].split()[0]
                    fasta_len[current_record] = 0
                else:
                    fasta_len[current_record] += len(line.strip())

        _gtf_chromosomes = set(window_annotate[GTF_CHROMOSOME].unique())
        _fasta_chromsomes = set(fasta_len.keys())
        _gtf_fasta_match = _gtf_chromosomes.intersection(_fasta_chromsomes)

        if len(_gtf_fasta_match) != len(_gtf_chromosomes):
            _msg = "GTF File Chromosomes {g} do not match FASTA File Chromosomes {f}\n"
            _msg += "The following chromosomes will not map correctly: {ft}"
            _msg = _msg.format(g=_gtf_chromosomes,
                               f=_fasta_chromsomes,
                               ft=_gtf_chromosomes.symmetric_difference(_fasta_chromsomes))
            warnings.warn(_msg)

        if len(_gtf_fasta_match) == 0:
            raise ValueError("Unable to map FASTA and GTF chromosomes together")

        for chromosome in _gtf_fasta_match:
            _chrlen = fasta_len[chromosome]
            _idx = window_annotate[GTF_CHROMOSOME] == chromosome
            window_annotate.loc[_idx & (window_annotate[SEQ_STOP] > _chrlen), SEQ_STOP] = _chrlen
            window_annotate.loc[_idx & (window_annotate[SEQ_START] > _chrlen), SEQ_START] = _chrlen

    return window_annotate


def _fix_genes(gene_dataframe):
    """
    Find minimum start and maximum stop
    :param gene_dataframe: pd.DataFrame
    :return:
    """

    # Make sure that the strandedness doesn't reverse start/stop
    assert (gene_dataframe[SEQ_START] <= gene_dataframe[SEQ_STOP]).all()

    def _most_common(x):
        return x.value_counts().index[0]

    # Define the functions for aggregating gene records
    aggregate_functions = {SEQ_START: min, SEQ_STOP: max, GTF_CHROMOSOME: _most_common, GTF_STRAND: _most_common}

    return gene_dataframe.groupby("gene_name").aggregate(aggregate_functions).reset_index()


def _add_TSS(gene_dataframe):
    """
    Add a TSS column in place
    :param gene_dataframe: pd.DataFrame
    :return:
    """
    gene_dataframe[SEQ_TSS] = gene_dataframe[SEQ_START].copy()
    rev_strand = gene_dataframe[GTF_STRAND] == "-"
    gene_dataframe.loc[rev_strand, SEQ_TSS] = gene_dataframe.loc[rev_strand, SEQ_STOP].copy()
    return gene_dataframe
