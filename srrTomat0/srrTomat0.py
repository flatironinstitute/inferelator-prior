import argparse
import os

import pandas as pd

from srrTomat0.processor.matrix import pileup_raw_counts
from srrTomat0.processor.srr import get_srr_files, unpack_srr_files
from srrTomat0.processor.star import star_align_fastqs
from srrTomat0.processor.utils import file_path_abs

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
    output_path = file_path_abs(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Download all the SRR files
    os.makedirs(os.path.join(output_path, SRR_SUBPATH), exist_ok=True)
    srr_file_names = get_srr_files(srr_ids, os.path.join(output_path, SRR_SUBPATH))

    # Unpack all the SRR files into FASTQ files
    os.makedirs(os.path.join(output_path, FASTQ_SUBPATH), exist_ok=True)
    fastq_file_names = unpack_srr_files(srr_ids, srr_file_names, os.path.join(output_path, FASTQ_SUBPATH))

    # Run all the FASTQ files through STAR to align and count genes
    os.makedirs(os.path.join(output_path, STAR_ALIGNMENT_SUBPATH), exist_ok=True)
    count_file_names = star_align_fastqs(srr_ids, fastq_file_names, star_reference_genome,
                                         os.path.join(output_path, STAR_ALIGNMENT_SUBPATH))

    # Convert the count files into a matrix and save it to a TSV
    count_matrix = pileup_raw_counts(srr_ids, count_file_names)
    count_file_name = os.path.join(output_path, OUTPUT_COUNT_FILE_NAME)

    # Save the raw counts file
    if gzip_output:
        count_matrix.to_csv(count_file_name + ".gz", compression='gzip', sep="\t")
    else:
        count_matrix.to_csv(count_file_name, sep="\t")
