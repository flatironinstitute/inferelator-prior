from __future__ import print_function

import os

import pandas as pd

from inferelator_prior import (SRR_SUBPATH, FASTQ_SUBPATH, STAR_ALIGNMENT_SUBPATH, HTSEQ_ALIGNMENT_SUBPATH,
                               KALLISTO_ALIGNMENT_SUBPATH)
from inferelator_prior.processor.htseq_count import htseq_count_aligned
from inferelator_prior.processor.matrix import pileup_raw_counts, normalize_matrix_to_fpkm, normalize_matrix_to_tpm
from inferelator_prior.processor.srr import get_srr_files, unpack_srr_files
from inferelator_prior.processor.star import star_align_fastqs
from inferelator_prior.processor.kallisto import kallisto_align_fastqs, KALLISTO_COUNT_COL, KALLISTO_TPM_COL
from inferelator_prior.processor.utils import file_path_abs, test_requirements_exist, ArgParseTestRequirements

OUTPUT_COUNT_FILE_NAME = "srr_counts.tsv"
OUTPUT_COUNT_METADATA_NAME = "srr_alignment_metadata.tsv"
OUTPUT_FPKM_FILE_NAME = "srr_fpkm.tsv"
OUTPUT_TPM_FILE_NAME = "srr_tpm.tsv"


def main():
    ap = ArgParseTestRequirements(description="Turn a list of RNAseq expression SRRs from NCBI GEO into a count matrix")
    ap.add_argument("-s", "--srr", dest="srr", help="SRR record IDs", nargs="+", metavar="SRRID", default=None)
    ap.add_argument("-f", "--file", dest="file", help="List of SRR records in a TXT file", metavar="FILE", default=None)
    ap.add_argument("-g", "--genome", dest="genome", help="Reference genome (STAR or Kallisto)", metavar="PATH",
                    required=True)
    ap.add_argument("-a", "--annotation", dest="anno", help="GTF/GFF Annotation File", metavar="FILE", default=None)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    ap.add_argument("--gzip", dest="gzip", help="GZIP output file", action='store_const', const=True, default=False)
    ap.add_argument("--cpu", dest="cpu", help="NUM of cores to use", metavar="NUM", type=int, default=4)
    ap.add_argument("--star_jobs", dest="sjob", help="NUM of STAR workers to use", metavar="NUM", type=int, default=4)
    ap.add_argument("--kallisto", dest="kallisto", help="Align and quant with Kallisto", action='store_const',
                    const=True, default=False)
    ap.add_argument("--skip_srr", dest="skip", help="Skip downloading & unpacking SRRs", action='store_const',
                    const=True, default=False)

    args, star_args = ap.parse_known_args()

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

    srr_tomat0(srr_ids, args.out, args.genome, annotation_file=args.anno, gzip_output=args.gzip, cores=args.cpu,
               star_jobs=args.sjob, star_args=star_args, kallisto=args.kallisto, skip=args.skip)


def srr_tomat0(srr_ids, output_path, star_reference_genome, annotation_file=None, gzip_output=False, cores=4, star_jobs=2,
               star_args=None, kallisto=False, skip=False):

    output_path = file_path_abs(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Download all the SRR files
    print("Downloading SRR files")
    os.makedirs(os.path.join(output_path, SRR_SUBPATH), exist_ok=True)
    srr_file_names = get_srr_files(srr_ids, os.path.join(output_path, SRR_SUBPATH), num_workers=cores, skip=skip)

    # Unpack all the SRR files into FASTQ files
    print("Unpacking SRR files")
    os.makedirs(os.path.join(output_path, FASTQ_SUBPATH), exist_ok=True)
    fastq_file_names = unpack_srr_files(srr_ids, srr_file_names, os.path.join(output_path, FASTQ_SUBPATH),
                                        num_workers=cores, skip=skip)

    gz_extension = ".gz" if gzip_output else ""

    count_matrix_file_name = os.path.join(output_path, OUTPUT_COUNT_FILE_NAME) + gz_extension
    fpkm_file_name = os.path.join(output_path, OUTPUT_FPKM_FILE_NAME) + gz_extension
    tpm_file_name = os.path.join(output_path, OUTPUT_TPM_FILE_NAME) + gz_extension
    
    if kallisto:
        print("Aligning FASTQ files")
        os.makedirs(os.path.join(output_path, KALLISTO_ALIGNMENT_SUBPATH), exist_ok=True)
        count_file_names = kallisto_align_fastqs(srr_ids, fastq_file_names, star_reference_genome,
                                                 os.path.join(output_path, KALLISTO_ALIGNMENT_SUBPATH),
                                                 num_workers=cores)

        tpm_df = None
        count_matrix = None

        for sid, cf in zip(srr_ids, count_file_names):

            if cf is None:
                continue

            counts = pd.read_csv(cf, sep="\t", index_col=0)

            if tpm_df is None:
                tpm_df = pd.DataFrame(index=counts.index)

            if count_matrix is None:
                count_matrix = pd.DataFrame(index=counts.index)

            tpm_df[sid] = counts[KALLISTO_TPM_COL]
            count_matrix[sid] = counts[KALLISTO_COUNT_COL]

        tpm_df.T.to_csv(tpm_file_name, sep="\t")
        count_matrix.T.to_csv(count_matrix_file_name, sep="\t")

    else:

        # Run all the FASTQ files through STAR to align
        print("Aligning FASTQ files")
        star_args = [] if star_args is None else star_args

        os.makedirs(os.path.join(output_path, STAR_ALIGNMENT_SUBPATH), exist_ok=True)
        thread_count = max(int(cores / len(srr_ids)), int(cores / star_jobs))
        sam_file_names = star_align_fastqs(srr_ids, fastq_file_names, star_reference_genome,
                                           os.path.join(output_path, STAR_ALIGNMENT_SUBPATH),
                                           num_workers=star_jobs, threads_per_worker=thread_count,
                                           star_options=star_args)

        # Run all the SAM files through HTSeq.count to count
        print("Counting SAM alignments")
        os.makedirs(os.path.join(output_path, HTSEQ_ALIGNMENT_SUBPATH), exist_ok=True)
        count_file_names = htseq_count_aligned(srr_ids, sam_file_names, annotation_file,
                                               os.path.join(output_path, HTSEQ_ALIGNMENT_SUBPATH), num_workers=cores)

        # Convert the count files into a matrix and save it to a TSV
        print("Assembling result matrix")
        count_matrix, count_metadata = pileup_raw_counts(srr_ids, count_file_names)

        # Save the raw counts file
        count_matrix.to_csv(count_matrix_file_name, sep="\t")

        # Save the count metadata file
        count_metadata.to_csv(os.path.join(output_path, OUTPUT_COUNT_METADATA_NAME), sep="\t")

        # Normalize to FPKM
        print("Normalizing result matrix to FPKM")
        normalized_count_matrix_fpkm = normalize_matrix_to_fpkm(count_matrix, annotation_file)

        # Save the normalized counts file
        normalized_count_matrix_fpkm.to_csv(fpkm_file_name, sep="\t")

        # Normalize to TPM
        print("Normalizing result matrix to TPM")
        normalized_count_matrix_tpm = normalize_matrix_to_tpm(count_matrix, annotation_file)

        # Save the normalized counts file
        normalized_count_matrix_tpm.to_csv(tpm_file_name, sep="\t")

    print("Count file {sh} generated from {srlen} SRA files".format(sh=count_matrix.shape, srlen=len(srr_ids)))
    failed_counts = list(map(lambda x: x is None, count_file_names))

    if any(failed_counts):
        print("{n} Sequence Records could not be counted:".format(n=sum(failed_counts)), end="")
        print("\n\t".join([sid for sid, fail in zip(srr_ids, failed_counts) if fail]))

    return count_matrix


if __name__ == '__main__':
    main()
