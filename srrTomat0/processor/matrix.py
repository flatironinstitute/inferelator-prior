import HTSeq
import pandas as pd

INDEX_NAME = "gene"
COUNT_COLUMN = "count"

META_STARTSWITH_FLAG = "__"
META_ALIGNED_COUNTS = "aligned_feature_sum"


# Turn count files into a count matrix
# TODO: test this
def pileup_raw_counts(srr_ids, count_files):
    """
    Convert the STAR alignment GeneCount files to a dataframe of SRR-derived expression values

    :param srr_ids: list(str)
        NCBI SRR ID string
    :param count_files: list(str)
        A list of HTSeq count files
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

        # Find the metadata
        count_metadata_indexes = count_data.index[count_data.index.str.startswith(META_STARTSWITH_FLAG)]

        # Process metadata
        count_meta_data = count_data.loc[count_metadata_indexes, :].rename(columns={COUNT_COLUMN: srr_id}).transpose()
        count_meta_data.columns = count_meta_data.columns.str.strip(META_STARTSWITH_FLAG)

        # Remove metadata from count dataframe
        count_data = count_data.drop(count_metadata_indexes)

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

        # Add the total counts to the metadata
        count_meta_data[META_ALIGNED_COUNTS] = count_data[COUNT_COLUMN].sum()
        meta_data.append(count_meta_data)

    # Combine the meta_data into a single dataframe
    meta_data = pd.concat(meta_data)

    return matrix_data, meta_data


# Turn a raw read count into a normalized RPKM / FPKM per gene
def normalize_matrix_to_fpkm(matrix_data, annotation_file):
    gff_reader = HTSeq.GFF_Reader(annotation_file)
    gene_lengths = pd.DataFrame.from_dict({gf.name: _gene_length(gf) for gf in gff_reader if gf.type == "gene"},
                                          orient='index', columns=['length'])

    diff = matrix_data.index.difference(gene_lengths.index)
    if len(diff) > 0:
        print("Gene lengths unknown for: {genes}".format(genes=" ".join(diff.tolist())))

    gene_lengths = gene_lengths.reindex(matrix_data.index)


    # Normalize the libraries by read depth
    normalized_matrix = matrix_data.divide(matrix_data.sum(), axis=1) * 10e6

    # Normalize the libraries by gene length
    normalized_matrix = normalized_matrix.divide(gene_lengths, axis=0)

    return normalized_matrix


def _gene_length(htseq_genomic_feature):
    return abs(htseq_genomic_feature.iv.start - htseq_genomic_feature.iv.end)
