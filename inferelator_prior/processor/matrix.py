import HTSeq
import pandas as pd

INDEX_NAME = "gene"
COUNT_COLUMN = "count"

META_STARTSWITH_FLAG = "__"
META_ALIGNED_COUNTS = "aligned_feature_sum"

TRANSCRIPT_TYPE_FLAG = "exon"


# Turn count files into a count matrix
# TODO: test this
def pileup_raw_counts(srr_ids, count_files):
    """
    Convert the HTSeq count files to a dataframe of SRR-derived expression values

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
        matrix_data[srr_id] = count_data[COUNT_COLUMN].astype(int)

        # Add the total counts to the metadata
        count_meta_data[META_ALIGNED_COUNTS] = count_data[COUNT_COLUMN].sum()
        meta_data.append(count_meta_data)

    # Combine the meta_data into a single dataframe
    meta_data = pd.concat(meta_data)

    return matrix_data, meta_data


# Turn a raw read count into a normalized RPKM / FPKM per gene
def normalize_matrix_to_fpkm(matrix_data, annotation_file):
    """
    Convert a raw count dataframe to a library and gene size normalized dataframe (RPKM / FPKM)

    :param matrix_data: pd.DataFrame [Genes x Samples]
        Dataframe of raw counts per gene
    :param annotation_file: str
        Path to the genome annotation (GTF) file
    :return normalized_matrix: pd.DataFrame [Genes x Samples]
        Normalized dataframe (FPKM)
    """

    gene_lengths = load_gene_lengths(annotation_file)

    diff = matrix_data.index.difference(gene_lengths.index)
    if len(diff) > 0:
        print("Dropping genes with unknown lengths: {genes}".format(genes=" ".join(diff.tolist())))

    normalized_matrix = matrix_data.drop(diff, axis=0)

    # Normalize the libraries by read depth to counts per million reads
    normalized_matrix = normalized_matrix.divide(normalized_matrix.sum()) * 1e6

    # Normalize the libraries by gene length to counts per kilobase per million reads
    normalized_matrix = normalized_matrix.divide(gene_lengths['length'], axis=0)

    return normalized_matrix


# Turn a raw read count into a normalized TPM per gene
def normalize_matrix_to_tpm(matrix_data, annotation_file):
    """
    Convert a raw count dataframe to a library and gene size normalized dataframe (TPM)

    :param matrix_data: pd.DataFrame [Genes x Samples]
        Dataframe of raw counts per gene
    :param annotation_file: str
        Path to the genome annotation (GTF) file
    :return normalized_matrix: pd.DataFrame [Genes x Samples]
        Normalized dataframe (TPM)
    """

    gene_lengths = load_gene_lengths(annotation_file)

    diff = matrix_data.index.difference(gene_lengths.index)
    if len(diff) > 0:
        print("Dropping genes with unknown lengths: {genes}".format(genes=" ".join(diff.tolist())))

    # Align data
    normalized_matrix = matrix_data.drop(diff, axis=0)
    gene_lengths = gene_lengths.reindex(normalized_matrix.index)

    # Normalize the libraries by gene length to counts per kilobase
    normalized_matrix = normalized_matrix.divide(gene_lengths['length'], axis=0)

    # Normalize the libraries by scaling to the library size
    normalized_matrix = normalized_matrix.divide(normalized_matrix.sum()) * 1e6

    return normalized_matrix


def load_gene_lengths(annotation_file):
    """
    Load gene lengths from an annotation file

    :param annotation_file: str
        Path to the genome annotation (GTF) file
    :return gene_lengths: pd.DataFrame[G x 1]
        Dataframe indexed by gene name

        ==========  ======= ==============================================================
        length      int     sum of exon length in kilobases
        ==========  ======= ==============================================================

    """

    # Load a GFF reader from HTSeq
    gff_reader = HTSeq.GFF_Reader(annotation_file)

    # Get exons for each gene
    gene_lengths = {}
    for gf in gff_reader:
        if gf.type == TRANSCRIPT_TYPE_FLAG:
            try:
                gene_lengths[gf.name].append(_gene_length(gf))
            except KeyError:
                gene_lengths[gf.name] = [_gene_length(gf)]

    # Sum exon lengths and pack into a dataframe in kilobases
    gene_lengths = pd.DataFrame.from_dict({gn: sum(exons) / 10e3 for gn, exons in gene_lengths.items()},
                                          orient='index', columns=['length'])

    return gene_lengths

def _gene_length(htseq_genomic_feature):
    """
    Get feature length
    :param htseq_genomic_feature: HTSeq.GenomeFeature
        GenomeFeature from a GFF_Reader iterable
    :return: int
        Feature length
    """
    return abs(htseq_genomic_feature.iv.start - htseq_genomic_feature.iv.end)
