from inferelator_prior.network_from_motifs import (add_common_arguments, parse_common_arguments,
                                                   load_and_process_motifs, scan_and_build_by_tf)
from inferelator_prior.processor.gtf import GTF_GENENAME, get_fasta_lengths, select_genes
from inferelator_prior.motifs.motif_scan import MotifScan

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Create a prior from a genome, TF motifs, and an optional BED file")

    # REQUIRED ARGUMENTS ###############################################################################################

    ap.add_argument("-m", "--motif", dest="motif", help="Motif file", metavar="PATH", required=True)
    ap.add_argument("-f", "--fasta", dest="fasta", help="Promoter FASTA file", metavar="FILE", required=True)
    #ap.add_argument("--scan-threshold", dest="scan_threshold", default="1e-4", help="Threshold for FIMO")

    # Process common arguments into values
    add_common_arguments(ap)
    args = ap.parse_args()

    out_prefix, _window, _tandem, _use_tss, _gl, _tfl, _minfo, _ = parse_common_arguments(args)

    _, _, prior_data = build_motif_prior_from_fasta(args.motif, args.fasta,
                                                    num_cores=args.cores,
                                                    scanner_thresh=args.scan_thresh,
                                                    motif_ic=args.min_ic,
                                                    tandem=_tandem,
                                                    scanner_type=args.scanner,
                                                    motif_format=args.motif_format,
                                                    output_prefix=out_prefix,
                                                    gene_constraint_list=_gl,
                                                    regulator_constraint_list=_tfl,
                                                    debug=args.debug,
                                                    fuzzy_motif_names=args.fuzzy,
                                                    motif_info=_minfo,
                                                    shuffle=args.shuffle,
                                                    save_locs=args.save_locs,
                                                    save_locs_filtered=args.save_locs_filtered)


def build_motif_prior_from_fasta(motif_file, promoter_fasta_file, scanner_type='fimo', num_cores=1, motif_ic=6,
                                 tandem=100, truncate_prob=0.35, scanner_thresh="1e-4", motif_format="meme",
                                 gene_constraint_list=None, regulator_constraint_list=None,
                                 output_prefix=None, debug=False, fuzzy_motif_names=False, motif_info=None,
                                 shuffle=None, save_locs=False, save_locs_filtered=False):
    """
    Build a motif-based prior from promoter sequences extracted into a FASTA.

    :param motif_file: Path to motif file (meme or transfac format)
    :type motif_file: str
    :param promoter_fasta_file: Path to FASTA file containing promoter DNA
    :type promoter_fasta_file: str
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
    :param gene_constraint_list: A file
    :type gene_constraint_list: str, None
    :param regulator_constraint_list:
    :type regulator_constraint_list: str, None
    :param output_prefix: Prefix output files with this path / prefix
    :type output_prefix: str
    :param debug:
    :type debug: bool
    :param fuzzy_motif_names: Use fuzzy merging of motif names
    :type fuzzy_motif_names: bool
    :param motif_info: Information about motifs. This will override any details read in from motif file.
        A template can be generated from ``inferelator_prior.motif_information`` and modified as needed.
    :type motif_info: pd.DataFrame
    :param shuffle: Randomly shuffle motif PWMs using this seed. None disables. Defaults to None.
    :type shuffle: None, int
    :return prior_matrix, raw_matrix, prior_data: Filtered connectivity matrix, unfiltered score matrix, and unfiltered
        long dataframe with scored TF->Gene pairs and genomic locations
    :rtype: pd.DataFrame, pd.DataFrame, pd.DataFrame
    """

    # PROCESS MOTIF PWMS ###############################################################################################

    motifs, motif_information = load_and_process_motifs(motif_file, motif_format, truncate_prob=truncate_prob,
                                                        regulator_constraint_list=regulator_constraint_list,
                                                        fuzzy=fuzzy_motif_names, motif_constraint_info=motif_info,
                                                        shuffle=shuffle)

    # SCAN PROMOTERS FOR MOTIFS ########################################################################################

    # Load and scan target chromatin peaks
    print("Scanning promoter sequences ({f_c}) for motifs ({f_m})".format(f_c=promoter_fasta_file, f_m=motif_file))

    MotifScan.set_type(scanner_type)

    promoters = get_fasta_lengths(promoter_fasta_file)
    genes = pd.DataFrame({GTF_GENENAME: list(promoters.keys())})

    if gene_constraint_list is not None:
        genes = select_genes(genes, gene_constraint_list)

    return scan_and_build_by_tf(promoter_fasta_file, None, genes, None, output_prefix, motif_information,
                                debug=debug, motif_ic=motif_ic, tandem=tandem, scanner_thresh=scanner_thresh,
                                save_locs=save_locs, save_locs_filtered=save_locs_filtered, num_cores=num_cores,
                                extract_genome=False, filter_motif_hits_for_gene_list=gene_constraint_list is not None, 
                                by_chromosome=False)


if __name__ == '__main__':
    main()
