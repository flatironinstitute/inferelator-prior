from srrTomat0.processor.gtf import GTF_CHROMOSOME, GTF_GENENAME, SEQ_START, SEQ_STOP
import pandas as pd

# Column names
BED_CHROMOSOME = 'chrom'
SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'


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
