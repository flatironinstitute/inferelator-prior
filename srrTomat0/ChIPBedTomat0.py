import argparse
import pandas as pd

import pybedtools

from srrTomat0.processor.gtf import load_gtf_to_dataframe, SEQ_START, SEQ_STOP, GTF_GENENAME, SEQ_TSS
from srrTomat0.processor.utils import file_path_abs
from srrTomat0.processor.bedtools import get_peaks_in_features

# Column names
BED_CHROMOSOME = 'chrom'
SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'

# Quantiles for bin
PEAK_QUANTILES = [0.25, 0.5, 0.75, 1]


def main():
    ap = argparse.ArgumentParser(description="Load peaks and genes.")
    ap.add_argument("-f", "--file", dest="file", help="TSV file with ID|BED PATH pairs", metavar="FILE", default=None)
    ap.add_argument("-b", "--bed", dest="bed", help="BED file containing ChIP peaks", nargs="+", metavar="FILE",
                    default=None)
    ap.add_argument("-a", "--annotation", dest="anno", help="GTF/GFF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output TSV PATH", metavar="PATH", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window size", type=int, default=0)
    ap.add_argument("-g", "--genebody", dest="gene", help="Gene body", action="store_const", const=True, default=False)
    ap.add_argument("-t", "--tss", dest="tss", help="Transcription start site", action="store_const", const=True,
                    default=False)
    args = ap.parse_args()

    if args.bed is None and args.file is None:
        print("One of --bed or --file must be set")
        exit(1)
    elif args.bed is not None and args.file is not None:
        print("Only one of --bed or --file may be set (not both)")
        exit(1)
    elif args.bed is not None:
        # SRR IDs are provided at command line
        id_names = args.bed
        chip_bed_files = args.bed
    elif args.file is not None:
        # SRR IDs are in a .txt file; read them into a list
        chip_samples = pd.read_csv(args.file, sep="\t", index_col=None, header=None)
        if chip_samples.shape[1] != 2:
            print("The TSV file must have two columns: ID and File_Path")
        id_names = chip_samples.iloc[:, 0].tolist()
        chip_bed_files = chip_samples.iloc[:, 1].tolist()
    else:
        raise ValueError("There is something wrong with this switch")

    if args.gene is False and args.tss is False:
        print("One of --genebody or --tss must be set")
        exit(1)
    elif args.gene is not False and args.tss is not False:
        print("Only one of --genebody or --tss may be set (not both)")
        exit(1)

    chip_bed_tomat0(id_names, chip_bed_files, args.anno, output_path=args.out, window_size=args.window_size,
                    gene_body_flag=args.gene, tss_flag=args.tss)


def chip_bed_tomat0(id_names, chip_peaks_file, annotation_file, output_path=None, window_size=0, gene_body_flag = False,
                    tss_flag = False):
    """
    Process a BED file of peaks into a integer peak-count matrix
    :param chip_peaks_file: list(str)
        List of paths to a BED file
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
    annotation_file = file_path_abs(annotation_file)

    # Load annotations into a dataframe with pybedtools
    # Adjust the start and stop positions to account for a flanking window
    genes = load_gtf_to_dataframe(annotation_file)

    if gene_body_flag:
        genes = open_window(genes, window_size)
    if tss_flag:
        genes = open_tss(genes, window_size)


    prior_data = pd.DataFrame(index=genes[GTF_GENENAME])
    for id_name, peak_file in zip(id_names, chip_peaks_file):
        # Load BED file into a dataframe with pybedtools
        peak_file = file_path_abs(peak_file)
        chip_peaks = pybedtools.BedTool(peak_file).to_dataframe()
        gene_counts = get_peaks_in_features(genes, chip_peaks)

        # Get non-zero quantiles and use them to bin peak overlap by length
        quantiles = gene_counts.loc[gene_counts[SEQ_COUNTS] != 0, SEQ_COUNTS].quantile(PEAK_QUANTILES)
        gene_counts[SEQ_BIN] = 0

        for i, qval in enumerate(quantiles.sort_values(ascending=True)):
            gene_counts.loc[gene_counts[SEQ_COUNTS] >= qval, SEQ_BIN] = i + 1

        # Rename the column with ID and reindex for join
        gene_counts = gene_counts.rename({SEQ_BIN: id_name}).set_index(GTF_GENENAME).drop([SEQ_COUNTS], axis=1)
        prior_data = prior_data.join(gene_counts, on=[GTF_GENENAME])

    if output_path is not None:
        prior_data.to_csv(output_path, sep="\t")

    return prior_data


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

def open_tss(annotation_file, window_size):
    """
    This needs to adjust the start and stop in the annotation dataframe with window sizes
    :param annotation_dataframe: pd.DataFrame
    :param window_size: int
    :return windowed_dataframe: pd.DataFrame
    """
    tss_dataframe = annotation_file.copy()
    tss_dataframe[SEQ_START] = tss_dataframe[SEQ_TSS] - window_size
    tss_dataframe[SEQ_STOP] = tss_dataframe[SEQ_TSS] + window_size
    tss_dataframe.loc[tss_dataframe[SEQ_START] < 0, SEQ_START] = 0
    return tss_dataframe

if __name__ == '__main__':
    main()
