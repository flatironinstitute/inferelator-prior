import pandas as pd

INDEX_NAME = "gene"
COUNT_COLUMN = "count"

META_STARTSWITH_FLAG = "__"


# Turn count files into a count matrix
# TODO: test this
def pileup_raw_counts(srr_ids, count_files):
    """
    Convert the STAR alignment GeneCount files to a dataframe of SRR-derived expression values

    :param aligned_data: dict
        A dict of STAR count files that's keyed by SRR ID
    :return matrix_data: pd.DataFrame [Genes x Samples]
        A dataframe of raw, unnormalized count values from all SRR alignments
    """
    matrix_data = pd.DataFrame()
    meta_data = []
    for srr_id, count_file_name in zip(srr_ids, count_files):

        if count_file_name is None:
            continue

        # Load in the count data
        count_data = pd.read_csv(count_file_name, sep="\t", index_col=0, header=None)
        count_data.index.name = INDEX_NAME
        count_data.columns = [COUNT_COLUMN]

        # Pull off the metadata
        count_metadata_indexes = count_data.index.str.startswith(META_STARTSWITH_FLAG)
        meta_data.append(count_data.loc[count_metadata_indexes, :].rename(columns={COUNT_COLUMN: srr_id}).transpose())
        count_data = count_data.drop(count_metadata_indexes, errors="ignore")

        # Make sure that the pileup matrix has all the genes
        if len(count_data.index.difference(matrix_data.index)) > 0:
            new_index = matrix_data.index.union(count_data.index)
            matrix_data = matrix_data.reindex(new_index)
            matrix_data[pd.isna(matrix_data)] = 0

        # Make sure that the count data is aligned to the matrix data
        if not count_data.index.equals(matrix_data.index):
            count_data = count_data.reindex(matrix_data.index)
            count_data[pd.isna(count_data)] = 0

        # Stick the count data onto the data frame
        count_data = count_data.reindex(matrix_data.index)
        matrix_data[srr_id] = count_data[COUNT_COLUMN]

    # Combine the meta_data into a single dataframe
    meta_data = pd.concat(meta_data)

    return matrix_data, meta_data


# Turn a raw read count into a normalized FPKM per gene
# TODO: make this a thing
def normalize_matrix_to_fpkm(matrix_data):
    normalized_matrix = pd.DataFrame()
    return normalized_matrix
