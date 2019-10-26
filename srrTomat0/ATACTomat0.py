import argparse
import os

import pandas as pd

from srrTomat0.processor.utils import file_path_abs
from srrTomat0.processor.srr import get_srr_files, unpack_srr_files
from srrTomat0.processor.star import star_align_fastqs
from srrTomat0.processor.samtools import sam_sort

from srrTomat0 import SRR_SUBPATH, FASTQ_SUBPATH, STAR_ALIGNMENT_SUBPATH, BAM_SUBPATH


OUTPUT_MATRIX_FILE_NAME = "atac_matrix.tsv"

COUNT_FILE_METAINDEXES = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]
COUNT_FILE_HEADER = ["Total", "MinusStrand", "PlusStrand"]
COUNT_FILE_HEADER_FOR_OUTPUT = "Total"


def main():
    ap = argparse.ArgumentParser(description="Turn ATAC-seq expression SRRs from NCBI GEO into a prior matrix")
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

    atac_tomat0(srr_ids, args.out, args.genome, gzip_output=args.gzip)


def atac_tomat0(srr_ids, output_path, star_reference_genome, gzip_output=False,  cores=4, star_jobs=2, star_args=None,
                min_quality=None):

    star_args = [] if star_args is None else star_args

    output_path = file_path_abs(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Download all the SRR files
    print("Downloading SRR files")
    os.makedirs(os.path.join(output_path, SRR_SUBPATH), exist_ok=True)
    srr_file_names = get_srr_files(srr_ids, os.path.join(output_path, SRR_SUBPATH), num_workers=cores)

    # Unpack all the SRR files into FASTQ files
    print("Unpacking SRR files")
    os.makedirs(os.path.join(output_path, FASTQ_SUBPATH), exist_ok=True)
    fastq_file_names = unpack_srr_files(srr_ids, srr_file_names, os.path.join(output_path, FASTQ_SUBPATH),
                                        num_workers=cores)

    # Run all the FASTQ files through STAR to align
    print("Aligning FASTQ files")
    os.makedirs(os.path.join(output_path, STAR_ALIGNMENT_SUBPATH), exist_ok=True)
    thread_count = max(int(cores / len(srr_ids)), int(cores / star_jobs))
    sam_file_names = star_align_fastqs(srr_ids, fastq_file_names, star_reference_genome,
                                       os.path.join(output_path, STAR_ALIGNMENT_SUBPATH),
                                       num_workers=star_jobs, threads_per_worker=thread_count, star_options=star_args)

    # Sort all the SAM files into BAM files
    print("Sorting SAM files into BAM files")
    os.makedirs(os.path.join(output_path, BAM_SUBPATH), exist_ok=True)
    bam_file_names = sam_sort(srr_ids, sam_file_names, os.path.join(output_path, BAM_SUBPATH), min_quality=min_quality,
                              num_workers=cores)


if __name__ == '__main__':
    main()


