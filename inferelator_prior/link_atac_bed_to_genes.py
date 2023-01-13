from inferelator_prior.processor.gtf import (
    load_gtf_to_dataframe
)

from inferelator_prior.processor.bedtools import (
    link_bed_to_genes
)

import argparse
import pathlib


def main():

    ap = argparse.ArgumentParser(
        description="Link ATAC peaks in a BED file to genes in a GTF file"
    )

    ap.add_argument(
        "-g",
        "--gtf",
        dest="annotation",
        help="GTF Annotation File",
        metavar="FILE",
        required=True
    )

    ap.add_argument(
        "-w",
        "--window",
        dest="window_size",
        help="Window around genes",
        type=int,
        default=1000,
        nargs="+"
    )

    ap.add_argument(
        "-b",
        "--bed",
        dest="bed",
        help="Peak BED file",
        required=True,
        nargs="+"
    )

    ap.add_argument(
        "--no_tss",
        dest="tss",
        help="Use gene body for window (not TSS)",
        action='store_const', const=False, default=True
    )
    ap.add_argument(
        "--no_intergenic",
        dest="no_intergenic",
        help="Drop peaks not linked to a gene",
        action='store_const', const=True, default=False
    )

    ap.add_argument(
        "-o",
        "--out",
        dest="out",
        help="Output BED",
        metavar="FILE",
        default="./peaks_to_genes.bed"
    )

    ap.add_argument(
        "-op",
        "--out_prefix",
        dest="op",
        help="Prefix for output file (if more than one is processed",
        metavar="PREFIX",
        default="joined_"
    )

    ap.add_argument(
        "--no_out_header",
        dest="out_header",
        help="Omit BEAD header in output file",
        action='store_const', const=False, default=True
    )

    args = ap.parse_args()

    genes = load_gtf_to_dataframe(args.annotation)
    print(f"{genes.shape[0]} genes loaded")

    if len(args.bed) == 1:
        link_bed_to_genes(
            args.bed[0],
            genes,
            args.out,
            use_tss=args.tss,
            window_size=args.window_size,
            non_gene_key=None if args.no_intergenic else "Intergenic",
            out_header=args.out_header
        )

    else:

        # Loop through all the BED files provided
        for bf in args.bed:
            new_out = pathlib.Path(bf)
            new_out = new_out.parent.joinpath(args.op + str(new_out.name))
            print(f"Processing {bf} (saving as {new_out})")
            link_bed_to_genes(
                bf,
                genes,
                new_out,
                use_tss=args.tss,
                window_size=args.window_size,
                non_gene_key=None if args.no_intergenic else "Intergenic",
                out_header=args.out_header
            )


if __name__ == '__main__':
    main()
