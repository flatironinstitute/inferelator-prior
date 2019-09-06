import argparse
import os

import pandas as pd
import pybedtools
import numpy as np

from srrTomat0.processor.utils import file_path_abs


def main():
    ap = argparse.ArgumentParser(description="Load peaks and genes.")
    ap.add_argument("-f", "--file", dest="file", help="bed file containing ChIP peaks", metavar="FILE", default=None)
    ap.add_argument("-a", "--annotation", dest="anno", help="GTF/GFF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)

    args = ap.parse_known_args()


def chip_tomat0(chip_peaks_file, output_path, annotation_file):
    output_path = file_path_abs(output_path)
    chip_peaks_file = file_path_abs(chip_peaks_file)
    annotation_file = file_path_abs(annotation_file)

    chip_peaks = pybedtools.BedTool(chip_peaks_file)
    chip_peaks = chip_peaks.to_dataframe()

    annotations = pybedtools.BedTool(annotation_file)
    annotations = annotations.to_dataframe()
    attributes = annotations['attributes'].str.extract('gene_id\s\"([A-Za-z0-9\.\-\(\)]+)\"\;', expand=False)
    annotations['gene_name'] = attributes
    genes = fix_genes(annotations[['seqname', 'start', 'end', 'gene_name']])

    genes = {val: df for val, df in genes.groupby('seqname')}
    chip_peaks = {val: df for val, df in chip_peaks.groupby('chrom')}

    gene_counts = {}
    for chromosome in set(chip_peaks.keys()).union(set(genes.keys())):

        try:
            chip_peaks[chromosome]
            genes[chromosome]
        except KeyError:
            continue

        def _find_overlap(x):
            return sum((x['start'] <= chip_peaks[chromosome]['end']) & (x['end'] >= chip_peaks[chromosome]['start']))

        gene_counts[chromosome] = genes[chromosome].apply(_find_overlap, axis=1)


def open_window(annotation_dataframe, window_size):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe:
    :param window_size:
    """
    pass


def fix_genes(gene_dataframe):
    """
    Find minimum start and maximum stop
    :return:
    """

    assert (gene_dataframe['start'] <= gene_dataframe['end']).all()
    return gene_dataframe.groupby("gene_name").aggregate({'start': min,
                                                          'end': max,
                                                          'seqname': lambda x:x.value_counts().index[0]}).reset_index()
