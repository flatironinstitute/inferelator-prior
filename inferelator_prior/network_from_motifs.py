from inferelator_prior.processor.gtf import (load_gtf_to_dataframe, open_window, GTF_CHROMOSOME,
                                             SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME)
from inferelator_prior.processor.prior import build_prior_from_motifs, summarize_target_per_regulator, MotifScorer
from inferelator_prior.motifs.motif_scan import MotifScan
from inferelator_prior.motifs import motifs_to_dataframe, INFO_COL, MOTIF_NAME_COL
from inferelator_prior.processor._species_constants import SPECIES_MAP, _DEFAULT_WINDOW, _DEFAULT_TANDEM

import argparse
import gc
import os
import pathlib
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Create a prior from a genome, TF motifs, and an optional BED file")
    ap.add_argument("-m", "--motif", dest="motif", help="Motif file", metavar="PATH", required=True)
    ap.add_argument("--motif_format", dest="motif_format", help="Motif file FORMAT (transfac or meme)",
                    metavar="FORMAT", default="meme")
    ap.add_argument("-b", "--bed", dest="atac", help="BED file", metavar="FILE", default=None)
    ap.add_argument("-f", "--fasta", dest="fasta", help="Genomic FASTA file", metavar="FILE", required=True)
    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH prefix", metavar="PATH", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=None, nargs="+")
    ap.add_argument("-c", "--cpu", dest="cores", help="Number of cores", metavar="CORES", type=int, default=None)
    ap.add_argument("--no_tss", dest="tss", help="Use gene body for window (not TSS)", action='store_const',
                    const=False, default=True)
    ap.add_argument("--scan", dest="scanner", help="FIMO or HOMER", type=str, default='fimo')
    ap.add_argument("--motif_preprocessing_ic", dest="min_ic", help="Minimum information content",
                    metavar="BITS", type=int, default=None)
    ap.add_argument("--tandem_window", dest="tandem", help="Bases between TF bindings to consider an array",
                    metavar="BASES", type=int, default=None)
    ap.add_argument("--threshold", nargs="+", default=None, type=str)
    ap.add_argument("--species", dest="species", help="Load settings for a target species. Overrides other settings",
                    default=None, type=str, choices=list(SPECIES_MAP.keys()) + [None])

    args = ap.parse_args()
    out_prefix = os.path.abspath(os.path.expanduser(args.out))
    out_path = os.path.join(*pathlib.PurePath(out_prefix).parts[:-1])
    if not os.path.exists(out_path):
        os.makedirs(out_prefix)

    _species = args.species.lower() if args.species is not None else None

    if _species is None:
        _window = args.window_size if args.window_size is not None else _DEFAULT_WINDOW
        _tandem = args.tandem if args.tandem is not None else _DEFAULT_TANDEM
        _use_tss = args.tss
    else:
        _window = SPECIES_MAP[_species]['window'] if args.window_size is None else args.window_size
        _tandem = SPECIES_MAP[_species]['tandem'] if args.tandem is None else args.tandem
        _use_tss = SPECIES_MAP[_species]['use_tss'] if args.tss else args.tss

    if args.threshold is None:

        if _use_tss:
            print("Scanning {w} around TSS for {t}bp TF arrays".format(w=_window, t=_tandem))
        else:
            print("Scanning {w} around gene for {t}bp TF arrays".format(w=_window, t=_tandem))

        prior_matrix, raw_matrix = build_atac_motif_prior(args.motif, args.atac, args.annotation,
                                                          args.fasta,
                                                          window_size=_window,
                                                          num_cores=args.cores,
                                                          use_tss=_use_tss,
                                                          motif_ic=args.min_ic,
                                                          tandem=_tandem,
                                                          scanner_type=args.scanner,
                                                          motif_format=args.motif_format,
                                                          output_prefix=out_prefix)

    else:
        motifs = MotifScan.load_motif_file(args.motif)
        motif_information = motifs_to_dataframe(motifs)
        motif_information = motif_information[[MOTIF_NAME_COL, INFO_COL]].groupby(MOTIF_NAME_COL).agg("max")

        edge_count = {}
        for t in args.threshold:
            prior_matrix, raw_matrix = build_atac_motif_prior(args.motif, args.atac, args.annotation,
                                                              args.fasta,
                                                              window_size=_window,
                                                              num_cores=args.cores,
                                                              use_tss=_use_tss, motif_ic=args.min_ic,
                                                              scanner_type=args.scanner,
                                                              scanner_thresh=t,
                                                              tandem=_tandem,
                                                              motif_format=args.motif_format)

            edge_count[t] = (raw_matrix != 0).sum(axis=0)

        edge_count = pd.concat(edge_count, axis=1)
        edge_count = edge_count.join(motif_information[INFO_COL])

        edge_count.to_csv(out_prefix + "_edge_count.tsv", sep="\t")


def build_atac_motif_prior(motif_file, atac_bed_file, annotation_file, genomic_fasta_file, window_size=0,
                           use_tss=True, scanner_type='fimo', num_cores=1, motif_ic=6, tandem=100,
                           truncate_motifs=0.35, scanner_thresh="1e-4", motif_format="meme",
                           gene_constraint_list_file=None, regulator_constraint_list_file=None, output_prefix=None):

    # Set the scanner type
    if scanner_type.lower() == 'fimo':
        MotifScan.set_type_fimo()
    elif scanner_type.lower() == 'homer':
        MotifScan.set_type_homer()
    else:
        raise ValueError("motif_type must be fimo or homer")

    # PROCESS GENE ANNOTATIONS #

    print("Loading genes from file ({f})".format(f=annotation_file))
    # Load genes and open a window
    genes = load_gtf_to_dataframe(annotation_file)
    print("\t{n} genes loaded".format(n=genes.shape[0]))

    # Constrain to a list of genes
    if gene_constraint_list_file is not None:
        _gene_constraint_list = pd.read_csv(gene_constraint_list_file, index_col=None).iloc[:, 0].tolist()
        _gene_constraint_list = list(map(lambda x: x.upper(), _gene_constraint_list))

        _gene_constraint_idx = genes[GTF_GENENAME].str.upper()
        _gene_constraint_idx = _gene_constraint_idx.isin(_gene_constraint_list)

        genes = genes.loc[_gene_constraint_idx, :].copy()
        print("{c} Genes Retained ({n} in constraint list)".format(c=genes.shape[0], n=len(_gene_constraint_list)))

    genes = open_window(genes, window_size=window_size, use_tss=use_tss, check_against_fasta=genomic_fasta_file)
    print("\tPromoter regions defined with window {w}".format(w=window_size))

    # PROCESS MOTIF PWMS #

    print("Loading motifs from file ({f})".format(f=motif_file))
    if motif_format.lower() == "meme":
        from inferelator_prior.motifs.meme import read
    elif motif_format.lower() == "transfac":
        from inferelator_prior.motifs.transfac import read
    elif motif_format.lower() == "homer":
        from inferelator_prior.motifs.homer_motif import read
    else:
        raise ValueError("motif_format must be 'meme', 'homer', or 'transfac'")

    motifs = read(motif_file)
    motif_information = motifs_to_dataframe(motifs)
    print("\t{n} motifs loaded".format(n=len(motif_information)))

    if regulator_constraint_list_file is not None:
        _regulator_constraint_list = pd.read_csv(regulator_constraint_list_file, index_col=None).iloc[:, 0].tolist()

    if truncate_motifs is not None:
        [x.truncate(threshold=truncate_motifs) for x in motifs]

    # SCAN CHROMATIN FOR MOTIFS #

    # Load and scan target chromatin peaks
    print("Scanning target chromatin ({f_c}) for motifs ({f_m})".format(f_c=atac_bed_file, f_m=motif_file))

    gene_locs = genes.loc[:, [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]].copy()
    gene_locs[[SEQ_START, SEQ_STOP]] = gene_locs[[SEQ_START, SEQ_STOP]].astype(int)

    motif_peaks = MotifScan.scanner(motifs=motifs, num_workers=num_cores).scan(genomic_fasta_file,
                                                                               atac_bed_file=atac_bed_file,
                                                                               promoter_bed=gene_locs,
                                                                               min_ic=motif_ic,
                                                                               threshold=scanner_thresh)

    # PROCESS CHROMATIN PEAKS INTO NETWORK MATRIX #

    # Process into an information score matrix
    print("Processing TF binding sites into prior")
    MotifScorer.set_information_criteria(min_binding_ic=motif_ic, max_dist=tandem)
    raw_matrix = summarize_target_per_regulator(genes, motif_peaks, motif_information, num_workers=num_cores)

    # Nuke a bunch of dataframes and force a cyclic check
    del motif_information, motif_peaks, genes, gene_locs, motifs
    gc.collect()
    print("{n} regulatory edges identified by motif search".format(n=(raw_matrix != 0).sum().sum()))

    if output_prefix is not None:
        print("Writing output file {o}".format(o=output_prefix + "_unfiltered_matrix.tsv.gz"))
        raw_matrix.to_csv(output_prefix + "_unfiltered_matrix.tsv.gz", sep="\t")

    # Choose edges to keep
    prior_matrix = build_prior_from_motifs(raw_matrix, num_workers=num_cores)
    print("Prior matrix with {n} edges constructed".format(n=prior_matrix.sum().sum()))

    if output_prefix is not None:
        print("Writing output file {o}".format(o=output_prefix + "_edge_matrix.tsv.gz"))
        (prior_matrix != 0).astype(int).to_csv(output_prefix + "_edge_matrix.tsv.gz", sep="\t")

    return prior_matrix, raw_matrix


if __name__ == '__main__':
    main()
