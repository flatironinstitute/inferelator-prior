import argparse
import os
import asyncio

import pandas as pd

from srrTomat0.processor.srr import get_srr_files_async, unpack_srr_file
from srrTomat0.processor.star import star_align_fastq
from srrTomat0.processor.matrix import pileup_raw_counts, normalize_matrix_to_fpkm

SRR_SUBPATH = "SRR"
FASTQ_SUBPATH = "FASTQ"
STAR_ALIGNMENT_SUBPATH = "STAR"

OUTPUT_COUNT_FILE_NAME = "srr_counts.tsv"
OUTPUT_FPKM_FILE_NAME = "srr_fpkm.tsv"


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
        # SRR IDs are provided at command line
        srr_ids = args.srr
    elif args.file is not None:
        # SRR IDs are in a .txt file; read them into a list
        srr_ids = pd.read_csv(args.file, sep="\t", index_col=None, header=None).iloc[:, 0].tolist()
    else:
        raise ValueError("There is something wrong with this switch")

    srr_tomat0(srr_ids, args.out, args.genome, gzip_output=args.gzip)


def srr_tomat0(srr_ids, output_path, star_reference_genome, gzip_output=False):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(output_path, exist_ok=True)

    # Download all the SRR files
    os.makedirs(os.path.join(output_path, SRR_SUBPATH), exist_ok=True)
    srr_file_names = get_srr_files_async(srr_ids, os.path.join(output_path, SRR_SUBPATH))

    os.makedirs(os.path.join(output_path, FASTQ_SUBPATH), exist_ok=True)
    os.makedirs(os.path.join(output_path, STAR_ALIGNMENT_SUBPATH), exist_ok=True)

    # For each SRR id, get the SRR file, unpack it to a fastq, and align it
    # Then save the path to the count file in a dict, keyed by SRR id
    aligned_data = {}
    for srr_id, srr_file_name in zip(srr_ids, srr_file_names):
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

