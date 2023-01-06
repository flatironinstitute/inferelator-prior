import gzip
import pandas as pd
import re

GTF_GENE_ID_REGEX = 'gene_id \"(.*?)\";'
GFF_GENE_ID_REGEX = '=gene:(.*?)[;|$]'

# Column names
GTF_ATTRIBUTES = 'attributes'
GTF_CHROMOSOME = 'seqname'
GTF_GENENAME = 'gene_name'
GTF_STRAND = 'strand'
SEQ_START = 'start'
SEQ_STOP = 'end'
SEQ_TSS = 'TSS'

WINDOW_UP = "Window_Start"
WINDOW_DOWN = "Window_End"

GTF_COLUMNS = [
    GTF_CHROMOSOME,
    "source",
    "feature",
    SEQ_START,
    SEQ_STOP,
    "score",
    "strand",
    "frame",
    "attributes"
]

GFF_FILETYPES = [
    ".gff",
    ".gff3",
    ".gff.gz",
    ".gff3.gz"
]

def _most_common(x):
    try:
        return x.value_counts().index[0]
    except IndexError:
        return pd.NA

# Define the functions for aggregating gene records
AGGREGATE_FUNCS = {
    SEQ_START: min,
    SEQ_STOP: max,
    GTF_CHROMOSOME: _most_common,
    GTF_STRAND: _most_common
}


def load_gtf_to_dataframe(
    gtf_path,
    annotations=None,
    fasta_record_lengths=None,
    gene_id_regex=None,
    additional_regex=None,
    rename_chromosome_dict=None
):
    """
    Loads genes from a GTF or GFF into a dataframe and returns them

    :param gtf_path: Path to the GTF or GFF file
    :type gtf_path: str
    :param fasta_record_lengths: A dict of valid FASTA records, keyed by name
    :type fasta_record_lengths: dict(int)
    :param gene_id_regex: Regular expression to extract gene ID,
        None autodetects, defaults to None
    :type gene_id_regex: str
    :param additional_regex: Dict of column: regex to extract values from
        annotations with regex and place into column, defaults to None
    :type additional_regex: dict, None
    :param rename_chromosome_dict: Replace the chromosome names with these
        values.
    :type rename_chromosome_dict: dict, None
    :return annotations: Loaded and processed gene annotations
        'gene_name': str
        'strand': str
        'start': int
        'end': int
        'seqname': str
    :rtype: pd.DataFrame
    """

    if gene_id_regex is not None:
        _gregex = gene_id_regex
    elif any(gtf_path.lower().endswith(x) for x in GFF_FILETYPES):
        _gregex = GFF_GENE_ID_REGEX
    else:
        _gregex = GTF_GENE_ID_REGEX

    aggregate_functions = AGGREGATE_FUNCS.copy()

    # Load annotations into a dataframe with pybedtools
    if annotations is None:
        annotations = pd.read_csv(
            gtf_path,
            sep="\t",
            names=GTF_COLUMNS,
            comment="#",
            low_memory=False
        )

    if len(annotations) == 0:
        raise ValueError("No records present in {f}".format(f=gtf_path))

    # Fix chromosome names to always be strings
    annotations[GTF_CHROMOSOME] = annotations[GTF_CHROMOSOME].astype(str)

    # Check chromosomes
    if fasta_record_lengths is not None:
        check_chromosomes_match(
            annotations,
            list(fasta_record_lengths.keys()),
            file_name=gtf_path
        )

    # Drop anything with NaNs which were probably comment lines
    annotations = annotations.dropna(subset=[SEQ_START, SEQ_STOP])

    # Regex extract the gene_id from the annotations column
    annotations[GTF_GENENAME] = annotations[GTF_ATTRIBUTES].str.extract(
        _gregex, expand=False, flags=re.IGNORECASE
    )

    if additional_regex is not None:
        for col, reg in additional_regex.items():
            annotations[col] = annotations[GTF_ATTRIBUTES].str.extract(
                reg, expand=False, flags=re.IGNORECASE
            )
            aggregate_functions[col] = _most_common

    # Drop any NaNs in GENE_NAME:
    annotations.dropna(inplace=True, subset=[GTF_GENENAME])

    if len(annotations) == 0:
        raise ValueError(
            "Unable to parse gene IDs from annotation file attributes"
        )

    # Define genes as going from the minimum start for any subfeature
    # to the maximum end for any subfeature
    annotations = _fix_genes(annotations, aggregate_functions)

    # Fix chromosome names based on lookup table
    if rename_chromosome_dict is not None:
        annotations = _rename_chromosomes(
            annotations,
            GTF_CHROMOSOME,
            rename_chromosome_dict,
            drop_unknowns=True
        )

    return _add_TSS(annotations)


def open_window(
    annotation_dataframe,
    window_size,
    use_tss=False,
    fasta_record_lengths=None,
    constrain_to_intergenic=False,
    include_entire_gene_body=False
):
    """
    Adjust the start and stop in the annotation dataframe
    with window sizes

    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :param use_tss: bool
    :param fasta_record_lengths:
    :return window_annotate: pd.DataFrame
    """

    new_df = annotation_dataframe.copy()

    new_df[WINDOW_UP], new_df[WINDOW_DOWN] = pd.NA, pd.NA

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
        _up_key = SEQ_TSS
        _down_key = SEQ_TSS

    else:
        _up_key = SEQ_START
        _down_key = SEQ_STOP

    _pos_idx = new_df[GTF_STRAND] == "+"
    _neg_idx = new_df[GTF_STRAND] == "-"

    for didx, u, d in [
        (_pos_idx, w_up, w_down),
        (_neg_idx, w_down, w_up)
    ]:

        new_df.loc[didx, WINDOW_UP] = (
            new_df.loc[didx, _up_key] - u
        ).astype(new_df.loc[didx, WINDOW_UP].dtype)

        new_df.loc[didx, WINDOW_DOWN] = (
            new_df.loc[didx, _down_key] + d
        ).astype(new_df.loc[didx, WINDOW_DOWN].dtype)

    new_df[WINDOW_UP] = new_df[WINDOW_UP].astype(int)
    new_df[WINDOW_DOWN] = new_df[WINDOW_DOWN].astype(int)

    new_df.loc[new_df[WINDOW_UP] < 1, WINDOW_UP] = 1

    if include_entire_gene_body:
        to_fix_pos = _pos_idx & (new_df[WINDOW_DOWN] < new_df[SEQ_STOP])
        to_fix_neg = _neg_idx & (new_df[WINDOW_UP] > new_df[SEQ_STOP])

        new_df.loc[to_fix_pos, WINDOW_DOWN] = new_df.loc[to_fix_pos, SEQ_STOP]
        new_df.loc[to_fix_neg, WINDOW_UP] = new_df.loc[to_fix_neg, SEQ_START]

    if fasta_record_lengths is not None:

        _gtf_fasta_match = set(new_df[GTF_CHROMOSOME].unique()).intersection(
            set(fasta_record_lengths.keys())
        )

        for chromosome in _gtf_fasta_match:
            _chrlen = fasta_record_lengths[chromosome]
            _idx = new_df[GTF_CHROMOSOME] == chromosome
            new_df.loc[_idx & (new_df[WINDOW_UP] > _chrlen), WINDOW_UP] = _chrlen
            new_df.loc[_idx & (new_df[WINDOW_DOWN] > _chrlen), WINDOW_DOWN] = _chrlen

    if constrain_to_intergenic:
        new_df = new_df.groupby(GTF_CHROMOSOME).apply(fix_overlap)
        new_df.reset_index(level=GTF_CHROMOSOME, inplace=True, drop=True)

    new_df[SEQ_START] = new_df[WINDOW_UP]
    new_df[SEQ_STOP] = new_df[WINDOW_DOWN]
    new_df.drop([WINDOW_UP, WINDOW_DOWN], axis=1, inplace=True)

    return new_df


def fix_overlap(dataframe):
    """
    Apply function that sets window start and stop positions so that they do
    not overlap with features

    :param dataframe: pd.DataFrame
    :return dataframe: pd.DataFrame
    """

    dataframe = dataframe.sort_values(by=SEQ_START)

    # Save a copy of the existing data before making changes
    windows = dataframe[[WINDOW_UP, WINDOW_DOWN]].copy()

    _stop_shift = dataframe[SEQ_STOP].shift(1)

    start_idx = dataframe[WINDOW_UP] < _stop_shift
    dataframe.loc[start_idx, WINDOW_UP] = _stop_shift.loc[start_idx].astype(
        dataframe[WINDOW_UP].dtype
    )

    _start_shift = dataframe[SEQ_START].shift(-1)

    stop_idx = dataframe[WINDOW_DOWN] > _start_shift
    dataframe.loc[stop_idx, WINDOW_DOWN] = _start_shift.loc[stop_idx].astype(
        dataframe[WINDOW_DOWN].dtype
    )

    # Undo checking for intergenic when there's a problem
    # usually it means overlapping genes, which I can't deal with
    bad_idx = dataframe[WINDOW_UP] >= dataframe[WINDOW_DOWN]
    replace_cols = [WINDOW_UP, WINDOW_DOWN]

    dataframe.loc[bad_idx, replace_cols] = windows.loc[bad_idx, replace_cols]

    return dataframe


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

    with _opener(fasta_file, mode="rt") as fasta_fh:
        current_record = None
        for line in fasta_fh:
            if line.startswith(">"):
                current_record = line[1:].split()[0]
                fasta_len[current_record] = 0
            elif line.startswith("#"):
                pass
            elif current_record is not None:
                fasta_len[current_record] += len(line.strip())
            else:
                raise ParseError(f"{fasta_file} is a malformed FASTA file")

    return fasta_len


def check_chromosomes_match(
    data_frame,
    chromosome_names,
    chromosome_column=GTF_CHROMOSOME,
    raise_no_overlap=True,
    file_name=None
):
    """
    Check and see if a list of chromosomes matches the unique chromsome
    names from a dataframe column

    :param data_frame: Dataframe with a column that has chromosome names
    :type data_frame: pd.DataFrame
    :param chromosome_names: A list of chromosome names to compare
    :type chromosome_names: list, set
    :param chromosome_column: The column in the dataframe with chromosomes
    :type chromosome_column: str
    :param raise_no_overlap: Raise a ValueError if the chromosomes don't
        match at all
    :type raise_no_overlap: bool
    :return: A list that is the intersection of the two chromosome lists
        (and therefore good)
    :rtype: list
    """

    _left_chr = set(data_frame[chromosome_column].unique())
    _right_chr = set(chromosome_names)
    _joint_match = _left_chr.intersection(_right_chr)

    if len(_joint_match) != len(_left_chr):

        _names_miss = _left_chr.symmetric_difference(_right_chr)
        _n_miss = len(_names_miss)

        _names_left = list(_left_chr)[0:min(len(_left_chr), 10, _n_miss)]
        _names_right = list(_right_chr)[0:min(len(_right_chr), 10, _n_miss)]

        print(
            f"File {file_name}: " if file_name is not None else ""
            f"Chromosomes {_names_left} "
            f"do not match Chromosomes {_names_right}\n"
            f"The following chromosomes will not map correctly: "
            f"{list(_names_miss)}"
        )

    if len(_joint_match) == 0 and raise_no_overlap:
        raise ValueError("No overlap between chromosomes together")

    return list(_joint_match)


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

    _n_constrain = len(gene_constraint_list)

    if _n_constrain == 0:
        raise ValueError(
            "No elements provided in gene_constraint_list"
        )

    _gene_constraint_list = list(
        map(
            lambda x: x.upper(),
            gene_constraint_list
        )
    )

    _gene_constraint_idx = gene_dataframe[GTF_GENENAME].str.upper()
    _gene_constraint_idx = _gene_constraint_idx.isin(_gene_constraint_list)

    if _gene_constraint_idx.sum() == 0:

        _n_genes = gene_dataframe.shape[0]

        raise ValueError(
            "No overlap between annotations ("
            f"{list(gene_dataframe[GTF_GENENAME][:min(3, _n_genes)])} "
            " ...) and constraint list ("
            f"{gene_constraint_list[:min(3, _n_constrain)]} ...)"
        )

    gene_dataframe = gene_dataframe.loc[_gene_constraint_idx, :].copy()

    print(
        f"{gene_dataframe.shape[0]} Genes Retained "
        f"({len(_gene_constraint_list)} in constraint list)"
    )

    return gene_dataframe


def _fix_genes(gene_dataframe, aggregate_functions):
    """
    Find minimum start and maximum stop
    :param gene_dataframe: pd.DataFrame
    :return:
    """

    # Make sure that the strandedness doesn't reverse start/stop
    assert (gene_dataframe[SEQ_START] <= gene_dataframe[SEQ_STOP]).all()

    return gene_dataframe.groupby(
        GTF_GENENAME
    ).aggregate(
        aggregate_functions
    ).reset_index()


def _add_TSS(gene_dataframe):
    """
    Add a TSS column in place

    :param gene_dataframe: pd.DataFrame
    :return:
    """
    gene_dataframe[SEQ_TSS] = gene_dataframe[SEQ_START]
    _rev = gene_dataframe[GTF_STRAND] == "-"

    gene_dataframe.loc[_rev, SEQ_TSS] = gene_dataframe.loc[_rev, SEQ_STOP]
    return gene_dataframe


def _rename_chromosomes(
    dataframe,
    column_to_rename,
    renamer_dict,
    drop_unknowns=False
):
    """
    Replace values in a dataframe column based on a lookup table

    :param dataframe: Dataframe
    :type dataframe: pd.DataFrame
    :param column_to_rename: Column to rename
    :type column_to_rename: str
    :param renamer_dict: Dict, keyed by original value, value to replace
    :type renamer_dict: dict
    :param drop_unknowns: Remove any values not in the renamer_dict,
        defaults to False
    :type drop_unknowns: bool, optional
    :return: Dataframe with the column changed
    :rtype: pd.DataFrame
    """

    renamed_col = dataframe[column_to_rename].copy()
    _is_replaced = pd.Series(False, index=renamed_col.index)

    for k in renamer_dict.keys():
        _kidx = renamed_col == k
        renamed_col[_kidx] = renamer_dict[k]
        _is_replaced |= _kidx

    dataframe[column_to_rename] = renamed_col

    if drop_unknowns and not _is_replaced.all():
        return dataframe.loc[_is_replaced, :]

    else:
        return dataframe


class ParseError(RuntimeError):
    pass
