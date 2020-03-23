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


def extract_bed_sequence(bed_file, genome_fasta, output_path=None):
    output_path = tempfile.gettempdir() if output_path is None else output_path
    output_file = os.path.join(output_path, os.path.split(genome_fasta)[1] + BEDTOOLS_EXTRACT_SUFFIX)

    if not isinstance(bed_file, pybedtools.BedTool):
        bed_file = pybedtools.BedTool(bed_file)

    try:
        bed_file.sequence(fi=genome_fasta, fo=output_file)
    except pybedtools.helpers.BEDToolsError as pbe:
        print(pbe.msg)

    return output_file


def load_bed_to_bedtools(bed):
    if bed is None:
        return None
    elif isinstance(bed, pd.DataFrame):
        return pybedtools.BedTool.from_dataframe(bed)
    else:
        return pybedtools.BedTool(bed)


def intersect_bed(*beds):

    if len(beds) == 1:
        return beds[0]

    beds = [b.sort() for b in beds]
    return beds[0].intersect(beds[1:], sorted=True)


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

