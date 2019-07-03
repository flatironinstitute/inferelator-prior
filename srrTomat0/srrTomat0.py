import argparse
import os

import pandas as pd

SRR_SUBPATH = "SRR"
FASTQ_SUBPATH = "FASTQ"
STAR_ALIGNMENT_SUBPATH = "STAR"

OUTPUT_COUNT_FILE_NAME = "srr_counts.tsv"
OUTPUT_FPKM_FILE_NAME = ""

COUNT_FILE_METAINDEXES = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]
COUNT_FILE_HEADER = ["Total", "MinusStrand", "PlusStrand"]
COUNT_FILE_HEADER_FOR_OUTPUT = "Total"


def main():
    ap = argparse.ArgumentParser(description="Turn a list of RNA-seq expression SRRs from NCBI GEO into a count matrix")
    ap.add_argument("-s", "--srr", dest="srr", help="SRR record IDs", nargs="+", metavar="SRRID", default=None)
    ap.add_argument("-f", "--file", dest="file", help="List of SRR records in a TXT file", metavar="FILE", default=None)
    ap.add_argument("-g", "--genome", dest="genome", help="STAR reference genome", metavar="PATH", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    ap.add_argument("--gzip", dest="gzip", help="GZIP output file", action='store_const', const=True, default=False)

    args = ap.parse_args()
    srr_ids = list()

    if args.srr is None and args.file is None:
        print("One of --srr or --file must be set")
        exit(1)
    elif args.srr is not None and args.file is not None:
        print("Only one of --srr or --file may be set (not both)")
        exit(1)
    elif args.srr is not None:
        srr_ids = args.srr
    elif args.file is not None:
        srr_ids = pd.read_csv(args.file, sep="\t", index_col=None, header=None).iloc[:, 0].tolist()
    else:
        raise ValueError("There is something wrong with this switch")

    srr_tomat0(srr_ids, args.out, args.genome, gzip_output=args.gzip)


def srr_tomat0(srr_ids, output_path, star_reference_genome, gzip_output=False):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(output_path, exist_ok=True)

    # For each SRR id, get the SRR file, unpack it to a fastq, and align it
    # Then save the path to the count file in a dict, keyed by SRR id
    aligned_data = {}
    for srr_id in srr_ids:
        srr_file_name = get_srr_file(srr_id,
                                     os.path.join(output_path, SRR_SUBPATH))
        fastq_file_names = unpack_srr_file(srr_id, srr_file_name,
                                           os.path.join(output_path, FASTQ_SUBPATH))
        count_file_name = star_align_fastq(srr_id, fastq_file_names, star_reference_genome,
                                           os.path.join(output_path, STAR_ALIGNMENT_SUBPATH))
        aligned_data[srr_id] = count_file_name

    # Turn the count files into a count matrix
    count_file_name = os.path.join(output_path, OUTPUT_COUNT_FILE_NAME)

    if os.path.exists(count_file_name):
        count_matrix = pd.read_csv(count_file_name, sep="\t")
    elif os.path.exists(count_file_name + ".gz"):
        count_matrix = pd.read_csv(count_file_name + ".gz", sep="\t")
    else:
        count_matrix = pileup_raw_counts(aligned_data)

        # Save the raw counts file
        if gzip_output:
            count_matrix.to_csv(count_file_name + ".gz", compression='gzip', sep="\t")
        else:
            count_matrix.to_csv(count_file_name, sep="\t")

    # Normalize the count matrix to FPKM and save it
    normalized_matrix = normalize_matrix_to_fpkm(count_matrix)

    # Save the fpkm counts file
    fpkm_file_name = os.path.join(output_path, OUTPUT_FPKM_FILE_NAME)
    if gzip_output:
        count_matrix.to_csv(fpkm_file_name + ".gz", compression='gzip', sep="\t")
    else:
        count_matrix.to_csv(fpkm_file_name, sep="\t")

    return normalized_matrix


# Download the SRR file from NCBI
# TODO: make this a thing
def get_srr_file(srr_id, target_path):
    """
    Take a SRR ID string and get the SRR file for it from NCBI. Raise a ValueError if it cannot be found

    :param srr_id: str
        NCBI SRR ID string
    :param target_path: str
        The path to put the SRR file
    :return srr_file_name: str
        The SRR file name (including path)
    """
    srr_file_name = ""
    return srr_file_name


# Unpack the SRR file to a fastQ file
# TODO: make this a thing
def unpack_srr_file(srr_id, srr_file_name, target_path):
    """

    :param srr_id: str
        NCBI SRR ID string
    :param srr_file_name: str
        The complete path to the SRR file
    :param target_path: str
        The path to put the FASTQ file(s)
    :return fastq_file_names: list
        A list of complete FASTQ file names that were unpacked from the SRR file (including path)
    """
    fastq_file_names = []
    return fastq_file_names


# Align and count the fastQ file with STAR
# TODO: make this a thing
def star_align_fastq(srr_id, fastq_file_names, reference_genome, output_path):
    """

    :param srr_id: str
        NCBI SRR ID string
    :param fastq_file_names: str
        A list of complete FASTQ file names that were unpacked from the SRR file (including path)
    :param reference_genome: str
        A path to the STAR reference genome that was preassembled
    :param output_path: str
        The path to put the output alignment files
    :return count_file_name: str
        The STAR count file generated by --quantMode (including path)
    """
    count_file_name = ""
    return count_file_name


# Turn count files into a count matrix
# TODO: test this
def pileup_raw_counts(aligned_data):
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
        count_metadata = count_data.loc[COUNT_FILE_METAINDEXES, :]
        count_data = count_data.drop(COUNT_FILE_METAINDEXES, errors="ignore")

        # Make sure that the pileup matrix has all the genes
        if len(count_data.index.difference(matrix_data.index)) > 0:
            matrix_data = matrix_data.reindex(count_data.index)
            matrix_data[pd.isna(matrix_data)] = 0

        # Make sure that the count data is aligned to the matrix data
        if not count_data.index.equals(matrix_data.index):
            count_data = count_data.reindex(matrix_data.index)

        # Stick the count data onto the data frame
        count_data = count_data.reindex(matrix_data.index)
        matrix_data[srr_id] = count_data[COUNT_FILE_HEADER_FOR_OUTPUT]

    return matrix_data


# Turn a raw read count into a normalized FPKM per gene
# TODO: make this a thing
def normalize_matrix_to_fpkm(matrix_data):
    normalized_matrix = pd.DataFrame()
    return normalized_matrix
