import pandas as pd
from srrTomat0.processor.star import STAR_COUNT_COLUMN, STAR_COUNT_FILE_METAINDEXES

# Turn count files into a count matrix
# TODO: test this
def pileup_raw_counts(aligned_data, count_meta_names=STAR_COUNT_FILE_METAINDEXES, count_column=STAR_COUNT_COLUMN):
    """
    Convert the STAR alignment GeneCount files to a dataframe of SRR-derived expression values

    :param aligned_data: dict
        A dict of STAR count files that's keyed by SRR ID
    :return matrix_data: pd.DataFrame [Genes x Samples]
        A dataframe of raw, unnormalized count values from all SRR alignments
    """
    matrix_data = pd.DataFrame()
    for srr_id, count_file_name in aligned_data.items():

        # Load in the count data
        count_data = pd.read_csv(count_file_name, sep="\t")

        # Pull off the metadata
        count_metadata_indexes = count_data.index.intersection(count_meta_names)
        count_metadata = count_data.loc[count_metadata_indexes, :]
        count_data = count_data.drop(count_metadata_indexes, errors="ignore")

        # Make sure that the pileup matrix has all the genes
        if len(count_data.index.difference(matrix_data.index)) > 0:
            matrix_data = matrix_data.reindex(count_data.index)
            matrix_data[pd.isna(matrix_data)] = 0

        # Make sure that the count data is aligned to the matrix data
        if not count_data.index.equals(matrix_data.index):
            count_data = count_data.reindex(matrix_data.index)

        # Stick the count data onto the data frame
        count_data = count_data.reindex(matrix_data.index)
        matrix_data[srr_id] = count_data[count_column]

    return matrix_data


# Turn a raw read count into a normalized FPKM per gene
# TODO: make this a thing
def normalize_matrix_to_fpkm(matrix_data):
    normalized_matrix = pd.DataFrame()
    return normalized_matrix