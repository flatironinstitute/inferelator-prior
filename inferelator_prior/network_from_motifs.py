from inferelator_prior.processor.gtf import (load_gtf_to_dataframe, open_window, select_genes, GTF_CHROMOSOME,
                                             SEQ_START, SEQ_STOP, GTF_STRAND)
from inferelator_prior.processor.prior import build_prior_from_motifs, summarize_target_per_regulator, MotifScorer
from inferelator_prior.motifs.motif_scan import MotifScan
from inferelator_prior.motifs import load_motif_file, select_motifs, truncate_motifs
from inferelator_prior.processor.species_constants import SPECIES_MAP, DEFAULT_WINDOW, DEFAULT_TANDEM

import argparse
import gc
import os
import pathlib
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Create a prior from a genome, TF motifs, and an optional BED file")

    # REQUIRED ARGUMENTS ###############################################################################################

    ap.add_argument("-m", "--motif", dest="motif", help="Motif file", metavar="PATH", required=True)
    ap.add_argument("-f", "--fasta", dest="fasta", help="Genomic FASTA file", metavar="FILE", required=True)

    # PROMOTER BED ARGUMENTS ###########################################################################################

    ap.add_argument("-p", "--promoter", dest="promoter", help="Promoter BED file", metavar="FILE", default=None)

    # GENE WINDOW ARGUMENTS ############################################################################################

    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="FILE", default=None)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=None, nargs="+")
    ap.add_argument("--no_tss", dest="tss", help="Use gene body for window (not TSS)", action='store_const',
                    const=False, default=True)

    # MOTIF ARGUMENTS ##################################################################################################

    ap.add_argument("--scan", dest="scanner", help="FIMO or HOMER", type=str, default='fimo')
    ap.add_argument("--motif_preprocessing_ic", dest="min_ic", help="Minimum information content",
                    metavar="BITS", type=int, default=None)
    ap.add_argument("--tandem_window", dest="tandem", help="Bases between TF bindings to consider an array",
                    metavar="BASES", type=int, default=None)
    ap.add_argument("--motif_format", dest="motif_format", help="Motif file FORMAT (transfac or meme)",
                    metavar="FORMAT", default="meme")
    ap.add_argument("--species", dest="species", help="Load settings for a target species. Overrides other settings",
                    default=None, type=str, choices=list(SPECIES_MAP.keys()) + [None])

    # OTHER OPTIONAL ARGUMENTS #########################################################################################

    ap.add_argument("-b", "--bed", dest="constraint", help="Constraint BED file", metavar="FILE", default=None)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH prefix", metavar="PATH", default="./prior")
    ap.add_argument("-c", "--cpu", dest="cores", help="Number of cores", metavar="CORES", type=int, default=None)
    ap.add_argument("--genes", dest="gene_list", help="A list of genes to build connectivity matrix for. Optional.",
                    default=None, type=str)
    ap.add_argument("--tfs", dest="tf_list", help="A list of TFs to build connectivity matrix for. Optional.",
                    default=None, type=str)

    args = ap.parse_args()
    out_prefix = os.path.abspath(os.path.expanduser(args.out))

    # Create output path if necessary
    out_path = os.path.join(*pathlib.PurePath(out_prefix).parts[:-1])
    if not os.path.exists(out_path):
        os.makedirs(out_prefix)

    # Get default values for a species if provided
    _species = args.species.lower() if args.species is not None else None

    if _species is None:
        _window = args.window_size if args.window_size is not None else DEFAULT_WINDOW
        _tandem = args.tandem if args.tandem is not None else DEFAULT_TANDEM
        _use_tss = args.tss
    else:
        _window = SPECIES_MAP[_species]['window'] if args.window_size is None else args.window_size
        _tandem = SPECIES_MAP[_species]['tandem'] if args.tandem is None else args.tandem
        _use_tss = SPECIES_MAP[_species]['use_tss'] if args.tss else args.tss

    _do_genes, _do_promoters = args.annotation is not None, args.promoter is not None

    if _do_genes and _do_promoters:
        raise ValueError("Providing both a GTF file to -g and a promoter BED file to -p is not supported")

    elif _do_genes:
        build_motif_prior_from_genes(args.motif, args.annotation, args.fasta,
                                     constraint_bed_file=args.constraint,
                                     window_size=_window,
                                     num_cores=args.cores,
                                     motif_ic=args.min_ic,
                                     tandem=_tandem,
                                     scanner_type=args.scanner,
                                     motif_format=args.motif_format,
                                     output_prefix=out_prefix,
                                     gene_constraint_list_file=args.gene_list,
                                     regulator_constraint_list_file=args.tf_list)

    elif _do_promoters:
        raise ValueError("Gene promoter location is not supported yet")

    else:
        raise ValueError("Provide a GTF file to -g or a promoter BED file to -p")


def build_motif_prior_from_genes(motif_file, annotation_file, genomic_fasta_file, constraint_bed_file=None,
                                 window_size=0, use_tss=True, scanner_type='fimo', num_cores=1, motif_ic=6, tandem=100,
                                 truncate_prob=0.35, scanner_thresh="1e-4", motif_format="meme",
                                 gene_constraint_list_file=None, regulator_constraint_list_file=None,
                                 output_prefix=None):
    """
    Build a motif-based prior from windows around annotated genes.
    
    :param motif_file: Path to motif file (meme or transfac format)
    :type motif_file: str
    :param annotation_file: Path to GTF file containing gene annotations
    :type annotation_file: str
    :param genomic_fasta_file: Path to FASTA file containing genomic DNA
    :type genomic_fasta_file: str
    :param constraint_bed_file: Path to BED file constraining genomic sequence to specific locations
    :type constraint_bed_file: str
    :param window_size: Genomic region to keep around TSS or gene body. Symmetric if a single value.
        If two values are provided as a tuple, they are (upstream, downstream)
    :type window_size: int, (int, int)
    :param use_tss: Use the 5' end of the gene annotation as the TSS
    :type use_tss: bool
    :param scanner_type: Which motif scanner to use. Options are 'fimo' or 'homer'
    :type scanner_type: str
    :param num_cores: Number of cores to use for parallelization. None uses all cores.
    :type num_cores: int, None
    :param motif_ic: Minimum informational content required to keep a motif hit. Defaults to 6 bits
    :type motif_ic: numeric
    :param tandem: Maximum distance to consider two TF bindings a single tandem array
    :type tandem: int
    :param truncate_prob: Truncate probabilities that flank a PWM/PFM until this value is reached. None disables.
    :type truncate_prob: numeric, None
    :param scanner_thresh: Probability threshold for the motif scanner. Defaults to 1e-4
    :type scanner_thresh: numeric
    :param motif_format: File format for motif_file. Defaults to meme. Also supports transfac.
    :type motif_format: str
    :param gene_constraint_list_file: A file
    :type gene_constraint_list_file: str, None
    :param regulator_constraint_list_file:
    :type regulator_constraint_list_file: str, None
    :param output_prefix:
    :type output_prefix: str
    :return prior_matrix, raw_matrix:
    :rtype: pd.DataFrame, pd.DataFrame
    """

    # PROCESS GENE ANNOTATIONS #########################################################################################

    print("Loading genes from file ({f})".format(f=annotation_file))
    # Load genes and open a window
    genes = load_gtf_to_dataframe(annotation_file)
    print("{n} genes loaded".format(n=genes.shape[0]))

    # Constrain to a list of genes
    if gene_constraint_list_file is not None:
        _gene_constraint_list = pd.read_csv(gene_constraint_list_file, index_col=None).iloc[:, 0].tolist()
        genes = select_genes(genes, _gene_constraint_list)

    genes = open_window(genes, window_size=window_size, use_tss=use_tss, check_against_fasta=genomic_fasta_file)
    print("Promoter regions defined with window {w} around {g}".format(w=window_size, g="TSS" if use_tss else "gene"))

    # PROCESS MOTIF PWMS ###############################################################################################

    motifs, motif_information = load_and_process_motifs(motif_file, motif_format, truncate_prob=truncate_prob,
                                                        regulator_constraint_list_file=regulator_constraint_list_file)

    # SCAN CHROMATIN FOR MOTIFS ########################################################################################

    # Load and scan target chromatin peaks
    print("Scanning target chromatin ({f_c}) for motifs ({f_m})".format(f_c=constraint_bed_file, f_m=motif_file))

    # Create a fake bed file with the gene promoter
    gene_locs = genes.loc[:, [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]].copy()
    gene_locs[[SEQ_START, SEQ_STOP]] = gene_locs[[SEQ_START, SEQ_STOP]].astype(int)

    return network_scan_and_build(motifs, motif_information, genes, genomic_fasta_file,
                                  constraint_bed_file=constraint_bed_file, promoter_bed_file=gene_locs,
                                  scanner_type=scanner_type, scanner_thresh=scanner_thresh, num_cores=num_cores,
                                  motif_ic=motif_ic, tandem=tandem, output_prefix=output_prefix)


def load_and_process_motifs(motif_file, motif_format, regulator_constraint_list_file=None, truncate_prob=None):

    motifs, motif_information = load_motif_file(motif_file, motif_format)

    if regulator_constraint_list_file is not None:
        _regulator_constraint_list = pd.read_csv(regulator_constraint_list_file, index_col=None).iloc[:, 0].tolist()
        motifs = select_motifs(motifs, _regulator_constraint_list)

    truncate_motifs(motifs, truncate_prob)

    return motifs, motif_information


def network_scan_and_build(motifs, motif_information, genes, genomic_fasta_file, constraint_bed_file=None,
                           promoter_bed_file=None, scanner_type='fimo', num_cores=1, motif_ic=6, tandem=100,
                           scanner_thresh="1e-4", output_prefix=None):

    # Load and scan target chromatin peaks
    MotifScan.set_type(scanner_type)

    motif_peaks = MotifScan.scanner(motifs=motifs, num_workers=num_cores).scan(genomic_fasta_file,
                                                                               constraint_bed_file=constraint_bed_file,
                                                                               promoter_bed=promoter_bed_file,
                                                                               min_ic=motif_ic,
                                                                               threshold=scanner_thresh)

    # PROCESS CHROMATIN PEAKS INTO NETWORK MATRIX ######################################################################

    # Process into an information score matrix
    print("Processing TF binding sites into prior")
    MotifScorer.set_information_criteria(min_binding_ic=motif_ic, max_dist=tandem)
    raw_matrix = summarize_target_per_regulator(genes, motif_peaks, motif_information, num_workers=num_cores)

    # Nuke a bunch of dataframes and force a cyclic check
    del motif_peaks
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
