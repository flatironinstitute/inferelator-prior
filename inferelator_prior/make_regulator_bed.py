from inferelator_prior.processor.gtf import (load_gtf_to_dataframe, open_window, select_genes, GTF_CHROMOSOME,
                                             SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME, get_fasta_lengths)

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Create a BED file from a GTF file")

    ap.add_argument("-f", "--fasta", dest="fasta", help="Genomic FASTA file", metavar="FILE", required=True)
    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=None, nargs="+")
    ap.add_argument("--no_tss", dest="tss", help="Use gene body for window (not TSS)", action='store_const',
                    const=False, default=True)
    ap.add_argument("--intergenic", dest="intergenic", help="Only consider intergenic regions", action='store_const',
                    const=True, default=None)
    ap.add_argument("-o", "--out", dest="out", help="Output BED", metavar="FILE", default="./gene.bed")


    args = ap.parse_args()

    _intergenic = args.intergenic if args.intergenic is not None else False
    _use_tss = args.tss

    print("Loading genes from file ({f})".format(f=args.annotation))
    # Load genes and open a window
    fasta_gene_len = get_fasta_lengths(args.fasta)
    genes = load_gtf_to_dataframe(args.annotation, fasta_record_lengths=fasta_gene_len)
    print("{n} genes loaded".format(n=genes.shape[0]))


    _msg = "Promoter regions defined with window {w} around {g}".format(w=args.window_size, g="TSS" if _use_tss else "gene")
    _msg += " [Intergenic]" if _intergenic else ""
    print(_msg)

    genes = open_window(genes, window_size=args.window_size, use_tss=_use_tss, fasta_record_lengths=fasta_gene_len,
                        constrain_to_intergenic=_intergenic)

    # Create a fake bed file with the gene promoter
    gene_locs = genes.loc[:, [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME]].copy()
    gene_locs[[SEQ_START, SEQ_STOP]] = gene_locs[[SEQ_START, SEQ_STOP]].astype(int)
    gene_locs = gene_locs.sort_values(by=[GTF_CHROMOSOME, SEQ_START])

    gene_locs.to_csv(args.out, sep="\t", index=False)


if __name__ == '__main__':
    main()
