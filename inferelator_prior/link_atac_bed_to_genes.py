from inferelator_prior.processor.gtf import (load_gtf_to_dataframe, open_window, GTF_CHROMOSOME,
                                             SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME)
from inferelator_prior.processor.bedtools import load_bed_to_bedtools, intersect_bed

import argparse


def main():
    ap = argparse.ArgumentParser(description="Link ATAC peaks in a BED file to genes in a GTF file")

    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=None, nargs="+")
    ap.add_argument("-b", "--bed", dest="bed", help="Peak BED file", default=None)
    ap.add_argument("--no_tss", dest="tss", help="Use gene body for window (not TSS)", action='store_const',
                    const=False, default=True)
    ap.add_argument("-o", "--out", dest="out", help="Output BED", metavar="FILE", default="./peaks_to_genes.bed")


    args = ap.parse_args()
    link_bed_to_genes(args.bed, args.annotation, args.out, use_tss=args.tss, window_size=args.window_size)


def link_bed_to_genes(bed_file, gene_annotation_file, out_file, use_tss=True, window_size=1000, dprint=print):

    dprint("Loading genes from file ({f})".format(f=gene_annotation_file))
    # Load genes and open a window
    genes = load_gtf_to_dataframe(gene_annotation_file)
    dprint("{n} genes loaded".format(n=genes.shape[0]))


    _msg = "Promoter regions defined with window {w} around {g}".format(w=window_size, g="TSS" if use_tss else "gene")
    dprint(_msg)

    genes_window = open_window(genes, window_size=window_size, use_tss=use_tss, include_entire_gene_body=True)

    # Create a fake bed file with the gene promoter
    genes_window = genes.loc[:, [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME]].copy()
    genes_window[[SEQ_START, SEQ_STOP]] = genes_window[[SEQ_START, SEQ_STOP]].astype(int)
    genes_window = genes_window.sort_values(by=[GTF_CHROMOSOME, SEQ_START])

    gene_bed = load_bed_to_bedtools(genes_window)
    bed_locs = load_bed_to_bedtools(bed_file)

    intersect_assign = intersect_bed(gene_bed, bed_locs, wb=True).to_dataframe()
    intersect_assign.rename({'score': 'gene'}, axis=1, inplace=True)
    intersect_assign.to_csv(out_file, sep="\t", index=False, header=False)

    return bed_locs.count(), len(intersect_assign)

if __name__ == '__main__':
    main()
