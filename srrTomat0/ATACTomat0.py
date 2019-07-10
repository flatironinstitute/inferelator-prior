import argparse

import pandas as pd

SRR_SUBPATH = "SRR"
FASTQ_SUBPATH = "FASTQ"
STAR_ALIGNMENT_SUBPATH = "STAR"

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


def atac_tomat0(srr_ids, output_path, star_reference_genome, gzip_output=False):
    pass


if __name__ == '__main__':
    main()
