import gzip
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

GTF_COLUMNS = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]


def load_gtf_to_dataframe(gtf_path, fasta_record_lengths=None):
    """
    Loads genes from a GTF into a dataframe and returns them
    :param gtf_path: Path to the GTF file
    :type gtf_path: str
    :param fasta_record_lengths: A dict of valid FASTA records, keyed by name
    :type fasta_record_lengths: dict(int)
    :return annotations: Loaded and processed gene annotations
        'gene_name': str
        'strand': str
        'start': int
        'end': int
        'seqname': str
    :rtype: pd.DataFrame
    """

    # Load annotations into a dataframe with pybedtools
    annotations = pd.read_csv(gtf_path, sep="\t", names=GTF_COLUMNS, comment="#")

    # Fix chromosome names to always be strings
    annotations[GTF_CHROMOSOME] = annotations[GTF_CHROMOSOME].astype(str)

    # Check chromosomes
    if fasta_record_lengths is not None:
        check_chromosomes_match(annotations, list(fasta_record_lengths.keys()), file_name=gtf_path)

    # Drop anything with NaNs which were probably comment lines
    annotations = annotations.loc[~pd.isnull(annotations[SEQ_START]) & ~pd.isnull(annotations[SEQ_STOP]), :]

    # Regex extract the gene_id from the annotations column
    annotations[GTF_GENENAME] = annotations[GTF_ATTRIBUTES].str.extract(GENE_ID_REGEX, expand=False)

    # Define genes as going from the minimum start for any subfeature to the maximum end for any subfeature
    annotations = _fix_genes(annotations)

    return _add_TSS(annotations)


def open_window(annotation_dataframe, window_size, use_tss=False, fasta_record_lengths=None):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :param use_tss: bool
    :param fasta_record_lengths:
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

    if fasta_record_lengths is not None:

        _gtf_fasta_match = set(window_annotate[GTF_CHROMOSOME].unique()).intersection(set(fasta_record_lengths.keys()))

        for chromosome in _gtf_fasta_match:
            _chrlen = fasta_record_lengths[chromosome]
            _idx = window_annotate[GTF_CHROMOSOME] == chromosome
            window_annotate.loc[_idx & (window_annotate[SEQ_STOP] > _chrlen), SEQ_STOP] = _chrlen
            window_annotate.loc[_idx & (window_annotate[SEQ_START] > _chrlen), SEQ_START] = _chrlen

    return window_annotate


def get_fasta_lengths(fasta_file):
    """
    Get the lengths of each record in a fasta file
    :param fasta_file: Filename
    :type fasta_file: str
    :return: A dict of integers keyed by chromosome name
    :rtype: dict
    """

    fasta_len = {}

    _opener = gzip.open if fasta_file.endswith(".gz") else open

    with _opener(fasta_file, mode="r") as fasta_fh:
        current_record = None
        for line in fasta_fh:
            if line.startswith(">"):
                current_record = line[1:].split()[0]
                fasta_len[current_record] = 0
            else:
                fasta_len[current_record] += len(line.strip())

    return fasta_len


def check_chromosomes_match(data_frame, chromosome_names, chromosome_column=GTF_CHROMOSOME, raise_no_overlap=True,
                            file_name=None):
    """
    Check and see if a list of chromosomes matches the unique chromsome names from a dataframe column

    :param data_frame: Dataframe with a column that has chromosome names
    :type data_frame: pd.DataFrame
    :param chromosome_names: A list of chromosome names to compare
    :type chromosome_names: list, set
    :param chromosome_column: The column in the dataframe with chromosomes
    :type chromosome_column: str
    :param raise_no_overlap: Raise a ValueError if the chromosomes don't match at all
    :type raise_no_overlap: bool
    :return: A list that is the intersection of the two chromosome lists (and therefore good)
    :rtype: list
    """

    _gtf_chromosomes = set(data_frame[chromosome_column].unique())
    _fasta_chromsomes = set(chromosome_names)
    _gtf_fasta_match = _gtf_chromosomes.intersection(_fasta_chromsomes)

    if len(_gtf_fasta_match) != len(_gtf_chromosomes):
        _msg = "File {fn}: ".format(fn=file_name)
        _msg += "Chromosomes {g} do not match FASTA File Chromosomes {f}\n"
        _msg += "The following chromosomes will not map correctly: {ft}"
        _msg = _msg.format(g=_gtf_chromosomes,
                           f=_fasta_chromsomes,
                           ft=_gtf_chromosomes.symmetric_difference(_fasta_chromsomes))
        print(_msg)

    if len(_gtf_fasta_match) == 0 and raise_no_overlap:
        raise ValueError("Unable to map FASTA and GTF chromosomes together")

    return list(_gtf_fasta_match)


def select_genes(gene_dataframe, gene_constraint_list):
    """
    Keep only genes in a list. Case-insensitive.

    :param gene_dataframe: Dataframe of genes
    :type gene_dataframe: pd.DataFrame
    :param gene_constraint_list: List of genes to keep. None disables.
    :type gene_constraint_list: list[str], None
    :return:
    """

    if gene_constraint_list is None:
        return gene_dataframe

    if len(gene_constraint_list) == 0:
        raise ValueError("No elements provided in gene_constraint_list")

    _gene_constraint_list = list(map(lambda x: x.upper(), gene_constraint_list))

    _gene_constraint_idx = gene_dataframe[GTF_GENENAME].str.upper()
    _gene_constraint_idx = _gene_constraint_idx.isin(_gene_constraint_list)

    if _gene_constraint_idx.sum() == 0:
        _msg = "No overlap between annotations ({an} ...) and constraint list ({li} ...)"
        _msg = _msg.format(an=list(gene_dataframe[GTF_GENENAME][:min(3, gene_dataframe.shape[0])]),
                           li=gene_constraint_list[:min(3, len(gene_constraint_list))])
        raise ValueError(_msg)

    gene_dataframe = gene_dataframe.loc[_gene_constraint_idx, :].copy()
    print("{c} Genes Retained ({n} in constraint list)".format(c=gene_dataframe.shape[0], n=len(_gene_constraint_list)))

    return gene_dataframe


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
