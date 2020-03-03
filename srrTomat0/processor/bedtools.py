from srrTomat0.processor.gtf import GTF_CHROMOSOME, GTF_GENENAME, SEQ_START, SEQ_STOP, GTF_STRAND
import pandas as pd
import pybedtools
import os
import subprocess
import tempfile

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"

# Column names
BED_CHROMOSOME = 'chrom'

SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'
SEQ_SCORE = 'p-value'


def get_peaks_in_features(feature_dataframe, peak_dataframe, feature_group_column=GTF_CHROMOSOME,
                          peak_group_column=BED_CHROMOSOME):
    genes = feature_dataframe.copy()

    # Add counts (and set to 0)
    genes[SEQ_COUNTS] = 0

    # Group genes and peaks by chromosome

    genes = {val: df for val, df in genes.groupby(feature_group_column)}
    peaks = {val: df for val, df in peak_dataframe.groupby(peak_group_column)}

    chromosomes = set(genes.keys()).intersection(set(peaks.keys()))

    # Count overlaps on a per-chromosome basis
    gene_counts = []
    for chromosome in chromosomes:

        # Function to return the number of overlaps with peaks in `chip_peaks`
        # Iterates over genes from GTF data frame (using apply)
        def _find_overlap(x):
            start_bool = x[SEQ_START] <= peaks[chromosome][SEQ_STOP]
            stop_bool = x[SEQ_STOP] >= peaks[chromosome][SEQ_START]
            if sum(start_bool & stop_bool) == 0:
                return 0
            selected_peaks = peaks[chromosome].loc[start_bool & stop_bool, :].copy()
            selected_peaks.loc[selected_peaks[SEQ_START] < x[SEQ_START], SEQ_START] = x[SEQ_START]
            selected_peaks.loc[selected_peaks[SEQ_STOP] > x[SEQ_STOP], SEQ_STOP] = x[SEQ_STOP]
            return sum(selected_peaks[SEQ_STOP] - selected_peaks[SEQ_START])

        # Add a chromosome column and then process into an integer peak count
        genes[chromosome][feature_group_column] = chromosome
        genes[chromosome][SEQ_COUNTS] = genes[chromosome].apply(_find_overlap, axis=1)
        gene_counts.append(genes[chromosome])

    # Combine all
    gene_counts = pd.concat(gene_counts).reset_index().loc[:, [GTF_GENENAME, SEQ_COUNTS]]

    return gene_counts


def load_bed_to_dataframe(bed_file_path, **kwargs):
    """
    :param bed_file_path: str
    :return: pd.DataFrame
    """

    return pd.read_csv(bed_file_path, sep="\t", index_col=None, **kwargs)


def merge_overlapping_peaks(peak_dataframe, feature_group_column=None, chromosome_column=GTF_CHROMOSOME,
                            start_column=SEQ_START, end_column=SEQ_STOP, strand_column=GTF_STRAND,
                            score_columns=None, max_distance=-1):
    """
    :param peak_dataframe: pd.DataFrame
    :param feature_group_column: str
    :param chromosome_column: str
    :param start_column: str
    :param end_column: str
    :param score_columns: [(str, str)]
        A list of tuples (column_name, merge_function), where column_name is the bed file column and the merge_function
        is the bedtools merge option (like `distinct` or `max`)
    :return: pd.DataFrame
    """

    all_columns = [chromosome_column, start_column, end_column]
    all_merge_functions = []

    if strand_column is not None:
        all_columns.append(strand_column)
        all_merge_functions.append('distinct')

    if score_columns is not None:
        for (col, funstr) in score_columns:
            if col not in peak_dataframe.columns:
                raise ValueError("Unknown score column {col}".format(col=col))
            else:
                # Force the score column to be a string because pybedtools is :-(
                peak_dataframe[col] = peak_dataframe[col].astype(str)
                all_columns.append(col)
                all_merge_functions.append(funstr)

    # Make a list of the column names to handle with bedtools merge
    all_merge_columns = list(map(lambda x: x + 4, range(len(all_merge_functions))))

    if feature_group_column is None:
        # Reindex to remove any other columns
        peak_dataframe = peak_dataframe.reindex(all_columns, axis=1)
        return _merge_peaks_with_bedtools(peak_dataframe, all_merge_columns, all_merge_functions,
                                          max_distance=max_distance)

    else:
        # Add the grouping column to the list
        all_columns.append(feature_group_column)

        # Reindex to remove any other columns
        peak_dataframe = peak_dataframe.reindex(all_columns, axis=1)

        # Iterate through groups of feature_group_column
        merged_dataframe = list()

        for group, group_data in peak_dataframe.groupby(feature_group_column):
            # Merge peaks
            merged_peaks = _merge_peaks_with_bedtools(group_data.drop(feature_group_column, axis=1), all_merge_columns,
                                                      all_merge_functions, max_distance=max_distance)

            # Add the group name and store the dataframe in a list
            merged_peaks[feature_group_column] = group
            merged_dataframe.append(merged_peaks)

        merged_dataframe = pd.concat(merged_dataframe)

        return merged_dataframe


def extract_bed_sequence(bed_file, genome_fasta, output_path=None):
    output_path = tempfile.gettempdir() if output_path is None else output_path
    output_file = os.path.join(output_path, os.path.split(genome_fasta)[1] + BEDTOOLS_EXTRACT_SUFFIX)

    if os.path.exists(output_file):
        return output_file

    bedtools_command = ["bedtools", "getfasta", "-fi", genome_fasta, "-bed", bed_file, "-fo", output_file]
    proc = subprocess.run(bedtools_command)

    if int(proc.returncode) != 0:
        print("bedtools getfasta failed for {file} ({cmd})".format(file=bed_file, cmd=" ".join(bedtools_command)))
        try:
            os.remove(output_file)
        except FileNotFoundError:
            pass
        return None

    return output_file


def _merge_peaks_with_bedtools(merge_data, merge_columns, merge_function_names, max_distance=0):
    """
    :param merge_data: pd.DataFrame
    :param merge_columns: list(int)
    :param merge_function_names: list(str)
    :return:
    """

    assert len(merge_columns) == len(merge_function_names)
    assert len(merge_columns) + 3 == merge_data.shape[1]

    # Load the data into a BedTool object
    pbt_data = pybedtools.BedTool.from_dataframe(merge_data).sort()

    if len(merge_columns) > 0:
        # Merge the overlapping peaks
        pbt_data = pbt_data.merge(d=max_distance, c=merge_columns, o=merge_function_names).to_dataframe(max_distance)
    else:
        pbt_data = pbt_data.merge(d=max_distance).to_dataframe()

    pbt_data.columns = merge_data.columns
    return pbt_data
