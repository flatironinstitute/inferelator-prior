import pybedtools

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

    # Regex extract the gene_id from the annotations column
    annotations[GTF_GENENAME] = annotations[GTF_ATTRIBUTES].str.extract(GENE_ID_REGEX, expand=False)

    # Define genes as going from the minimum start for any subfeature to the maximum end for any subfeature
    annotations = _fix_genes(annotations)
    return _add_TSS(annotations)


def open_window(annotation_dataframe, window_size, use_tss=False):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :param use_tss: bool
    :return windowed_dataframe: pd.DataFrame
    """
    windowed_dataframe = annotation_dataframe.copy()

    if use_tss:
        windowed_dataframe[SEQ_START] = windowed_dataframe[SEQ_TSS] - window_size
        windowed_dataframe[SEQ_STOP] = windowed_dataframe[SEQ_TSS] + window_size
    else:
        windowed_dataframe[SEQ_START] = windowed_dataframe[SEQ_START] - window_size
        windowed_dataframe[SEQ_STOP] = windowed_dataframe[SEQ_STOP] + window_size

    windowed_dataframe.loc[windowed_dataframe[SEQ_START] < 0, SEQ_START] = 0

    return windowed_dataframe


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
