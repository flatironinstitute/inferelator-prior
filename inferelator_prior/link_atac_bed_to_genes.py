from inferelator_prior.processor.gtf import (load_gtf_to_dataframe, open_window, GTF_CHROMOSOME,
                                             SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME)
from inferelator_prior.processor.bedtools import (load_bed_to_bedtools, intersect_bed, load_bed_to_dataframe,
                                                  BED_CHROMOSOME)

import argparse
import pathlib
import pandas as pd

GENE_COL = "gene"
PEAK_COL = "peak"

def main():
    ap = argparse.ArgumentParser(description="Link ATAC peaks in a BED file to genes in a GTF file")

    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=None, nargs="+")
    ap.add_argument("-b", "--bed", dest="bed", help="Peak BED file", required=True, nargs="+")
    ap.add_argument("--no_tss", dest="tss", help="Use gene body for window (not TSS)", action='store_const',
                    const=False, default=True)
    ap.add_argument("--no_intergenic", dest="no_intergenic", help="Drop peaks not linked to a gene", action='store_const',
                    const=True, default=False)
    ap.add_argument("-o", "--out", dest="out", help="Output BED", metavar="FILE", default="./peaks_to_genes.bed")
    ap.add_argument("-op", "--out_prefix", dest="op", help="Prefix for output file (if more than one is processed",
                    metavar="PREFIX", default="joined_")
    ap.add_argument("--no_out_header", dest="out_header", help="Omit BEAD header in output file", action='store_const',
                    const=False, default=True)

    args = ap.parse_args()

    bed_files_for_linking = args.bed

    if len(bed_files_for_linking) == 1:
        link_bed_to_genes(args.bed, args.annotation, args.out, use_tss=args.tss, window_size=args.window_size,
                        non_gene_key=None if args.no_intergenic else "Intergenic", out_header=args.out_header)
    else:
        genes = load_gtf_to_dataframe(args.annotation)
        print("{n} genes loaded".format(n=genes.shape[0]))

        for bf in bed_files_for_linking:
            new_out = pathlib.Path(bf)
            new_out = new_out.parent.joinpath(args.op + str(new_out.name))
            print("Processing {f} (saving as {nf})".format(f=bf, nf=new_out))
            link_bed_to_genes(bf, genes, new_out, use_tss=args.tss, window_size=args.window_size,
                              non_gene_key=None if args.no_intergenic else "Intergenic", out_header=args.out_header)


def link_bed_to_genes(bed_file, gene_annotation_file, out_file, use_tss=True, window_size=1000, dprint=print,
                      non_gene_key="Intergenic", out_header=False):
    """
    Link a BED file (of arbitraty origin) to a set of genes from a GTF file based on proximity

    :param bed_file: Path to the BED file
    :type bed_file: str
    :param gene_annotation_file: Path to the genome annotation file (GTF)
    :type gene_annotation_file: str
    :param out_file: Path to the output file
    :type out_file: str
    :param use_tss: Base gene proximity on the TSS, not the gene body; defaults to True
    :type use_tss: bool, optional
    :param window_size: Window size (N, M) for proximity, where N is upstream of the gene and M is downstream. 
        If given as an integer K, interpreted as (K, K); defaults to 1000
    :type window_size: int, tuple, optional
    :param dprint: Debug message function (can be overridden to silence), defaults to print
    :type dprint: callable, optional
    :param non_gene_key: Name for BED peaks that aren't in the genome feature windows.
        Set to None to drop peaks that aren't in the genome feature windows; defaults to "Intergenic"
    :type non_gene_key: str, optional
    :return: Number of peaks before mapping, number of peaks after mapping, dataframe of peaks
    :rtype: int, int, pd.DataFrame
    """

    if isinstance(gene_annotation_file, str):
        dprint("Loading genes from file ({f})".format(f=gene_annotation_file))
        # Load genes and open a window
        genes = load_gtf_to_dataframe(gene_annotation_file)
        dprint("{n} genes loaded".format(n=genes.shape[0]))

    elif isinstance(gene_annotation_file, pd.DataFrame):
        genes = gene_annotation_file.copy()

    _msg = "Promoter regions defined with window {w} around {g}".format(w=window_size, g="TSS" if use_tss else "gene")
    dprint(_msg)

    genes_window = open_window(genes, window_size=window_size, use_tss=use_tss, include_entire_gene_body=True)

    # Create a fake bed file with the gene promoter
    genes_window = genes.loc[:, [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME]].copy()
    genes_window[[SEQ_START, SEQ_STOP]] = genes_window[[SEQ_START, SEQ_STOP]].astype(int)
    genes_window = genes_window.sort_values(by=[GTF_CHROMOSOME, SEQ_START])

    gene_bed = load_bed_to_bedtools(genes_window)

    # Load BED-type file to a dataframe
    # Explicitly cast chromosomes into strings (edge condition when chromosomes are just 1, 2, 3, Mt, etc)
    bed_df = load_bed_to_dataframe(bed_file)
    bed_df[BED_CHROMOSOME] = bed_df[BED_CHROMOSOME].astype(str)

    bed_locs = load_bed_to_bedtools(bed_df[[BED_CHROMOSOME, SEQ_START, SEQ_STOP]])

    try:
        ia = intersect_bed(gene_bed, bed_locs, wb=True).to_dataframe()
    except:
        print("Gene BED file:")
        print(gene_bed.to_dataframe().head())
        print("Target BED dataframe:")
        print(bed_df.head())
        print("Target BED file:")
        print(bed_locs.to_dataframe().head())
        raise

    ia.rename({'score': GENE_COL}, axis=1, inplace=True)

    # Rebuild an A/B bed file
    ia.columns = ['a_chrom', 'a_start', 'a_end', GTF_STRAND, GENE_COL, BED_CHROMOSOME, SEQ_START, SEQ_STOP]
    ia = ia[[BED_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND, GENE_COL]].copy()

    # Add an intergenic key if set; otherwise peaks that don't overlap will be dropped
    if non_gene_key is not None:
        ia = ia.merge(bed_df, how="outer", on=[BED_CHROMOSOME, SEQ_START, SEQ_STOP])
        ia[GENE_COL] = ia[GENE_COL].fillna(non_gene_key)
        ia[GTF_STRAND] = ia[GTF_STRAND].fillna("")
    
    # Make unique peak IDs based on gene
    ia[PEAK_COL] = ia[GENE_COL].groupby(
        ia[GENE_COL]
    ).transform(
        lambda x: pd.Series(map(lambda y: "_" + str(y), range(len(x))), index=x.index)
    )
    ia[PEAK_COL] = ia[GENE_COL].str.cat(ia[PEAK_COL]) 

    peaks = ia[PEAK_COL].copy()
    ia.drop(PEAK_COL, inplace=True, axis=1)
    ia.insert(5, PEAK_COL, peaks)

    # Sort for output
    ia = ia.sort_values(by=[BED_CHROMOSOME, SEQ_START]).reset_index(drop=True)


    if out_file is not None:
        ia.to_csv(out_file, sep="\t", index=False, header=out_header)

    return bed_locs.count(), len(ia), ia


if __name__ == '__main__':
    main()
