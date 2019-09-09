import argparse
import os

import pandas as pd
import pybedtools
import numpy as np

from srrTomat0.processor.utils import file_path_abs

GENE_ID_REGEX = 'gene_id\s\"([A-Za-z0-9\.\-\(\)]+)\"\;'

# Column names
GTF_ATTRIBUTES = 'attributes'
GTF_CHROMOSOME = 'seqname'
GTF_GENENAME = 'gene_name'
BED_CHROMOSOME = 'chrom'

SEQ_START = 'start'
SEQ_STOP = 'end'
SEQ_COUNTS = 'count'


def main():
    ap = argparse.ArgumentParser(description="Load peaks and genes.")
    ap.add_argument("-f", "--file", dest="file", help="bed file containing ChIP peaks", metavar="FILE", default=None)
    ap.add_argument("-a", "--annotation", dest="anno", help="GTF/GFF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output TSV PATH", metavar="PATH", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=0)

    args = ap.parse_args()
    chip_tomat0(args.file, args.out, args.anno, window_size=args.window_size)


def chip_tomat0(chip_peaks_file, annotation_file, output_path=None, window_size=0):
    """
    Process a BED file of peaks into a integer peak-count matrix
    :param chip_peaks_file: str
        Path to a BED file
    :param output_path: str
        Path to the output TSV file
    :param annotation_file: str
        Path to the GTF annotation file
    :param window_size: int
        Window on each side of a gene to include a peak in the count
        100 means 100bp up from start and 100bp down from end
    :return gene_counts: pd.DataFrame
        Integer count matrix of peaks per gene
    """

    # Convert paths to absolutes
    output_path = file_path_abs(output_path)
    chip_peaks_file = file_path_abs(chip_peaks_file)
    annotation_file = file_path_abs(annotation_file)

    # Load BED file into a dataframe with pybedtools
    chip_peaks = pybedtools.BedTool(chip_peaks_file).to_dataframe()

    # Load annotations into a dataframe with pybedtools
    annotations = pybedtools.BedTool(annotation_file).to_dataframe()

    # Regex extract the gene_id from the annotations column
    annotations[GTF_GENENAME] = annotations[GTF_ATTRIBUTES].str.extract(GENE_ID_REGEX, expand=False)
    genes = fix_genes(annotations[[GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_GENENAME]])

    # Adjust the start and stop positions to account for a flanking window
    genes = open_window(genes, window_size)

    # Add counts (and set to 0)
    genes[SEQ_COUNTS] = 0

    # Group genes and peaks by chromosome
    genes = {val: df for val, df in genes.groupby(GTF_CHROMOSOME)}
    chip_peaks = {val: df for val, df in chip_peaks.groupby(BED_CHROMOSOME)}

    # Count overlaps on a per-chromosome basis
    gene_counts = []
    for chromosome in set(chip_peaks.keys()).union(set(genes.keys())):

        if (chromosome not in chip_peaks) or (chromosome not in genes):
            continue  # Someone's using a weird chromosome name

        def _find_overlap(x):
            # Function to return the number of overlaps with peaks in `chip_peaks`
            # Iterates over genes from GTF data frame (using apply)
            start_bool = x[SEQ_START] <= chip_peaks[chromosome][SEQ_STOP]
            stop_bool = x[SEQ_STOP] >= chip_peaks[chromosome][SEQ_START]
            return sum(start_bool & stop_bool)

        genes[chromosome][GTF_CHROMOSOME] = chromosome
        genes[chromosome][SEQ_COUNTS] = genes[chromosome].apply(_find_overlap, axis=1)
        gene_counts.append(genes[chromosome])

    # Combine all
    gene_counts = pd.concat(gene_counts).reset_index().loc[:, [GTF_GENENAME, SEQ_COUNTS]]

    if output_path is not None:
        gene_counts.to_csv(output_path, sep="\t", index=False)

    return gene_counts


def open_window(annotation_dataframe, window_size):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :return windowed_dataframe: pd.DataFrame
    """
    windowed_dataframe = annotation_dataframe.copy()
    windowed_dataframe[SEQ_START] = windowed_dataframe[SEQ_START] - window_size
    windowed_dataframe[SEQ_STOP] = windowed_dataframe[SEQ_STOP] + window_size
    windowed_dataframe.loc[windowed_dataframe[SEQ_START] < 0, SEQ_START] = 0
    return windowed_dataframe


def fix_genes(gene_dataframe):
    """
    Find minimum start and maximum stop
    :return:
    """

    assert (gene_dataframe['start'] <= gene_dataframe['end']).all()
    return gene_dataframe.groupby("gene_name").aggregate({'start': min,
                                                          'end': max,
                                                          'seqname': lambda x:x.value_counts().index[0]}).reset_index()


if __name__ == '__main__':
    main()
