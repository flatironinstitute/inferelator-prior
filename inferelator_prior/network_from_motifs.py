from inferelator_prior.processor.gtf import (
    load_gtf_to_dataframe,
    open_window,
    select_genes,
    GTF_CHROMOSOME,
    SEQ_START,
    SEQ_STOP,
    GTF_STRAND,
    GTF_GENENAME,
    get_fasta_lengths,
)
from inferelator_prior.processor.prior import (
    build_prior_from_motifs,
    summarize_target_per_regulator,
    MotifScorer,
    PRIOR_TF,
    PRIOR_GENE,
)
from inferelator_prior.motifs.motif_scan import MotifScan
from inferelator_prior.motifs import (
    load_motif_file,
    select_motifs,
    truncate_motifs,
    fuzzy_merge_motifs,
    shuffle_motifs,
    MOTIF_COL,
    MOTIF_NAME_COL,
    MOTIF_OBJ_COL,
)
from inferelator_prior.processor.species_constants import (
    SPECIES_MAP,
    DEFAULT_WINDOW,
    DEFAULT_TANDEM,
)

import argparse
import os
import pathlib
import pathos
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(
        description="Create a prior from a genome, TF motifs, and a BED file"
    )

    # REQUIRED ARGUMENTS ######################################################
    ap.add_argument(
        "-m",
        "--motif",
        dest="motif",
        help="Motif file",
        metavar="PATH",
        required=True
    )
    ap.add_argument(
        "-f",
        "--fasta",
        dest="fasta",
        help="Genomic FASTA file",
        metavar="FILE",
        required=True,
    )

    # BED ARGUMENTS ###########################################################

    ap.add_argument(
        "-b",
        "--bed",
        dest="constraint",
        help="Constraint BED file",
        metavar="FILE",
        default=None,
    )

    # GENE WINDOW ARGUMENTS ###################################################

    ap.add_argument(
        "-g",
        "--gtf",
        dest="annotation",
        help="GTF Annotation File",
        metavar="FILE",
        default=None,
    )
    ap.add_argument(
        "-w",
        "--window",
        dest="window_size",
        help="Window around genes",
        type=int,
        default=None,
        nargs="+",
    )
    ap.add_argument(
        "--no_tss",
        dest="tss",
        help="Use gene body for window (not TSS)",
        action="store_const",
        const=False,
        default=True,
    )

    # Process common arguments into values
    add_common_arguments(ap)
    args = ap.parse_args()
    (
        out_prefix,
        _window,
        _tandem,
        _use_tss,
        _gl,
        _tfl,
        _minfo,
        _intergenic,
    ) = parse_common_arguments(args)

    prior_matrix, raw_matrix, prior_data = build_motif_prior_from_genes(
        args.motif,
        args.annotation,
        args.fasta,
        constraint_bed_file=args.constraint,
        use_tss=_use_tss,
        window_size=_window,
        num_cores=args.cores,
        motif_ic=args.min_ic,
        tandem=_tandem,
        scanner_type=args.scanner,
        scanner_thresh=args.scan_thresh,
        motif_format=args.motif_format,
        output_prefix=out_prefix,
        gene_constraint_list=_gl,
        regulator_constraint_list=_tfl,
        debug=args.debug,
        fuzzy_motif_names=args.fuzzy,
        motif_info=_minfo,
        shuffle=args.shuffle,
        lowmem=not args.highmem,
        intergenic_only=_intergenic,
        save_locs=args.save_locs,
        save_locs_filtered=args.save_locs_filtered,
    )

    print(f"Prior matrix with {prior_matrix.sum().sum()} edges constructed")


def add_common_arguments(argp):
    # MOTIF ARGUMENTS #########################################################

    argp.add_argument(
        "--scan",
        dest="scanner",
        help="FIMO or HOMER",
        type=str,
        default="FIMO",
        choices=["FIMO", "HOMER"],
    )
    argp.add_argument(
        "--scan-threshold",
        dest="scan_thresh",
        help="Scanner score threshold",
        type=str,
        default="1e-4",
    )
    argp.add_argument(
        "--motif_preprocessing_ic",
        dest="min_ic",
        help="Minimum information content",
        metavar="BITS",
        type=float,
        default=None,
    )
    argp.add_argument(
        "--tandem_window",
        dest="tandem",
        help="Bases between TF bindings to consider an array",
        metavar="BASES",
        type=int,
        default=None,
    )
    argp.add_argument(
        "--intergenic",
        dest="intergenic",
        help="Only consider intergenic regions",
        action="store_const",
        const=True,
        default=None,
    )
    argp.add_argument(
        "--motif_format",
        dest="motif_format",
        help="Motif file FORMAT (transfac or meme)",
        metavar="FORMAT",
        default="meme",
    )
    argp.add_argument(
        "--motif_info",
        dest="motif_info",
        help="Motif information TSV FILE",
        metavar="File",
        default=None,
    )
    argp.add_argument(
        "--species",
        dest="species",
        help="Load settings for a target species.",
        default=None,
        type=str,
        choices=list(SPECIES_MAP.keys()) + [None],
    )

    # OTHER OPTIONAL ARGUMENTS ################################################

    argp.add_argument(
        "-o",
        "--out",
        dest="out",
        help="Output PATH prefix",
        metavar="PATH",
        default="./prior",
    )
    argp.add_argument(
        "--save_location_data",
        dest="save_locs",
        help="Save a dataframe with TF->Gene binding locations",
        action="store_const",
        const=True,
        default=False,
    )
    argp.add_argument(
        "--save_filtered_location_data",
        dest="save_locs_filtered",
        help="Save a dataframe with post-filter TF->Gene binding locations",
        action="store_const",
        const=True,
        default=False,
    )
    argp.add_argument(
        "-c",
        "--cpu",
        dest="cores",
        help="Number of cores",
        metavar="CORES",
        type=int,
        default=None,
    )
    argp.add_argument(
        "--genes",
        dest="genes",
        help="A list of genes to build connectivity matrix for. Optional.",
        default=None,
        type=str,
    )
    argp.add_argument(
        "--tfs",
        dest="tfs",
        help="A list of TFs to build connectivity matrix for. Optional.",
        default=None,
        type=str,
    )
    argp.add_argument(
        "--debug",
        dest="debug",
        help="Activate Debug Mode",
        action="store_const",
        const=True,
        default=False,
    )
    argp.add_argument(
        "--fuzzy",
        dest="fuzzy",
        help="Use fuzzy motif name merging",
        action="store_const",
        const=True,
        default=False,
    )
    argp.add_argument(
        "--shuffle",
        dest="shuffle",
        help="Shuffle motif PWMs using SEED",
        metavar="SEED",
        const=42,
        default=None,
        action="store",
        nargs="?",
        type=int,
    )
    argp.add_argument(
        "--highmem",
        dest="highmem",
        help="Run in high memory mode (no performance benefits)",
        action="store_const",
        const=True,
        default=False,
    )


def parse_common_arguments(args):
    out_prefix = os.path.abspath(os.path.expanduser(args.out))

    # Create output path if necessary
    out_path = os.path.join(*pathlib.PurePath(out_prefix).parts[:-1])
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get default values for a species if provided
    _species = args.species.lower() if args.species is not None else None
    _intergenic = args.intergenic if args.intergenic is not None else False

    if _species is None:
        try:
            if args.window_size is None:
                _window = DEFAULT_WINDOW
            else:
                _window = args.window_size
            _use_tss = args.tss
        except AttributeError:
            _window, _use_tss = None, None

        _tandem = args.tandem if args.tandem is not None else DEFAULT_TANDEM
    else:
        try:
            _window = (
                SPECIES_MAP[_species]["window"]
                if args.window_size is None
                else args.window_size
            )

            if args.tss:
                _use_tss = SPECIES_MAP[_species]["use_tss"]
            else:
                _use_tss = args.tss
        except AttributeError:
            _window, _use_tss = None, None

        if args.tandem is None:
            _tandem = SPECIES_MAP[_species]["tandem"]
        else:
            _tandem = args.tandem

    # Load gene and regulator lists
    _gl = (
        pd.read_csv(args.genes, index_col=None, header=None)[0].tolist()
        if args.genes is not None
        else None
    )
    _tfl = (
        pd.read_csv(args.tfs, index_col=None, header=None)[0].tolist()
        if args.tfs is not None
        else None
    )

    # Load motif constraint info
    if args.motif_info is not None:
        _m = pd.read_csv(args.motif_info, sep="\t", index_col=None)
        _m.attrs["filename"] = args.motif_info
        if MOTIF_COL not in _m.columns or MOTIF_NAME_COL not in _m.columns:
            raise ValueError(
                f"motif_info must have columns {MOTIF_COL} and "
                f"{MOTIF_NAME_COL}; use inferelator_prior.motif_information "
                "as a template"
            )
    else:
        _m = None

    return out_prefix, _window, _tandem, _use_tss, _gl, _tfl, _m, _intergenic


def build_motif_prior_from_genes(
    motif_file,
    annotation_file,
    genomic_fasta_file,
    constraint_bed_file=None,
    window_size=0,
    use_tss=True,
    scanner_type="fimo",
    num_cores=1,
    motif_ic=6,
    tandem=100,
    truncate_prob=0.35,
    scanner_thresh="1e-4",
    motif_format="meme",
    gene_constraint_list=None,
    regulator_constraint_list=None,
    output_prefix=None,
    debug=False,
    fuzzy_motif_names=False,
    motif_info=None,
    shuffle=None,
    lowmem=True,
    intergenic_only=True,
    save_locs=False,
    save_locs_filtered=False,
):
    """
    Build a motif-based prior from windows around annotated genes.

    :param motif_file: Path to motif file (meme or transfac format)
    :type motif_file: str
    :param annotation_file: Path to GTF file containing gene annotations
    :type annotation_file: str
    :param genomic_fasta_file: Path to FASTA file containing genomic DNA
    :type genomic_fasta_file: str
    :param constraint_bed_file: Path to BED file constraining genomic sequence
        to specific locations
    :type constraint_bed_file: str
    :param window_size: Genomic region to keep around TSS or gene body.
        Symmetric if a single value.
        If two values are provided as a tuple, they are (upstream, downstream)
    :type window_size: int, (int, int)
    :param use_tss: Use the 5' end of the gene annotation as the TSS
    :type use_tss: bool
    :param scanner_type: Which motif scanner to use.
        Options are 'fimo' or 'homer'.
    :type scanner_type: str
    :param num_cores: Number of cores to use for parallelization.
        None uses all cores.
    :type num_cores: int, None
    :param motif_ic: Minimum informational content required to keep a
        motif hit. Defaults to 6 bits
    :type motif_ic: numeric
    :param tandem: Maximum distance to consider two TF bindings a
        single tandem array
    :type tandem: int
    :param truncate_prob: Truncate probabilities that flank a PWM/PFM
        until this value is reached. None disables.
    :type truncate_prob: numeric, None
    :param scanner_thresh: Probability threshold for the motif scanner.
        Defaults to 1e-4
    :type scanner_thresh: numeric
    :param motif_format: File format for motif_file. Defaults to meme.
        Also supports transfac.
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
    :param motif_info: Information about motifs. This will override
        any details read in from motif file. A template can be generated
        from ``inferelator_prior.motif_information`` and modified as needed.
    :type motif_info: pd.DataFrame
    :param shuffle: Randomly shuffle motif PWMs using this seed.
        None disables. Defaults to None.
    :type shuffle: None, int
    :param lowmem: Process TFs individually instead of all at once to
        minimize memory footprint
    :type lowmem: bool
    :param intergenic_only: Only scan intergenic regions for regulatory motifs
    :type intergenic_only: bool
    :param save_locs: Save motif mapping positions to a file, Defaults to False
    :type save_locs: bool
    :param save_locs_filtered: Save filtered TF -> Gene mapping locations to a
        file, Defaults to False
    :type save_locs_filtered: bool
    :return prior_matrix, raw_matrix, prior_data: Filtered connectivity matrix,
        unfiltered score matrix, and unfiltered long dataframe with scored
        TF->Gene pairs and genomic locations
    :rtype: pd.DataFrame, pd.DataFrame, pd.DataFrame
    """

    if save_locs and output_prefix is not None:
        save_locs = output_prefix + "_tf_binding_locs.tsv"

    if save_locs_filtered and output_prefix is not None:
        save_locs_filtered = output_prefix + "_tf_binding_locs_filtered.tsv"

    # PROCESS GENE ANNOTATIONS ################################################

    print("Loading genes from file ({f})".format(f=annotation_file))

    # Load genes and open a window

    fasta_gene_len = get_fasta_lengths(genomic_fasta_file)

    genes = load_gtf_to_dataframe(
        annotation_file,
        fasta_record_lengths=fasta_gene_len
    )

    print(f"{genes.shape[0]} genes loaded")

    # Constrain to a list of genes
    if gene_constraint_list is not None:
        genes = select_genes(genes, gene_constraint_list)

    genes = open_window(
        genes,
        window_size=window_size,
        use_tss=use_tss,
        fasta_record_lengths=fasta_gene_len,
        constrain_to_intergenic=intergenic_only,
    )

    print(
        f"Promoter regions defined with window {window_size} "
        f"around {'TSS' if use_tss else 'gene'}"
    )

    # PROCESS MOTIF PWMS ######################################################

    motifs, motif_information = load_and_process_motifs(
        motif_file,
        motif_format,
        truncate_prob=truncate_prob,
        regulator_constraint_list=regulator_constraint_list,
        fuzzy=fuzzy_motif_names,
        motif_constraint_info=motif_info,
        shuffle=shuffle,
    )

    # SCAN CHROMATIN FOR MOTIFS AND SCORE HITS ################################
    # Load and scan target chromatin peaks
    print(
        f"Scanning target chromatin ({constraint_bed_file}) "
        f"for motifs ({motif_file})"
    )

    MotifScan.set_type(scanner_type)

    # Create a fake bed file with the gene promoter
    gene_locs = genes.loc[
        :,
        [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]
    ].copy()

    gene_locs[
        [SEQ_START, SEQ_STOP]
    ] = gene_locs[[SEQ_START, SEQ_STOP]].astype(int)

    if not lowmem:
        raw_matrix, prior_data = network_scan(
            motifs,
            motif_information,
            genes,
            genomic_fasta_file,
            constraint_bed_file=constraint_bed_file,
            promoter_bed_file=gene_locs,
            scanner_type=scanner_type,
            scanner_thresh=scanner_thresh,
            num_cores=num_cores,
            motif_ic=motif_ic,
            tandem=tandem,
            debug=debug,
            save_locs=save_locs,
        )

        # PROCESS SCORES INTO NETWORK #########################################
        print(
            f"{(raw_matrix != 0).sum().sum()} regulatory edges "
            "identified by motif search"
        )

        return network_build(
            raw_matrix,
            prior_data,
            num_cores=num_cores,
            output_prefix=output_prefix,
            debug=debug,
            save_locs_filtered=save_locs_filtered,
        )

    else:
        return scan_and_build_by_tf(
            genomic_fasta_file,
            constraint_bed_file,
            genes,
            gene_locs,
            output_prefix,
            motif_information,
            debug=debug,
            motif_ic=motif_ic,
            tandem=tandem,
            scanner_thresh=scanner_thresh,
            save_locs=save_locs,
            save_locs_filtered=save_locs_filtered,
            num_cores=num_cores,
        )


def load_and_process_motifs(
    motif_file,
    motif_format,
    regulator_constraint_list=None,
    truncate_prob=None,
    fuzzy=False,
    motif_constraint_info=None,
    shuffle=None,
):
    motifs, motif_information = load_motif_file(motif_file, motif_format)

    if fuzzy:
        motif_information = fuzzy_merge_motifs(motif_information)
        motifs = motif_information[MOTIF_OBJ_COL].tolist()

    if shuffle:
        print("Shuffling motif PWMs")
        shuffle_motifs(motifs, random_seed=shuffle)

    if motif_constraint_info is not None:
        print(
            "Loaded info for {n} motifs from {f}".format(
                n=len(motif_constraint_info),
                f=motif_constraint_info.attrs.get("filename", None),
            )
        )

        _before_n = len(motifs)
        # Join the loaded motifs onto the provided info to decide what to keep
        mi_join = motif_information.set_index(
            MOTIF_COL
        ).drop(
            MOTIF_NAME_COL, axis=1
        )
        motif_information = motif_constraint_info[
            [MOTIF_COL, MOTIF_NAME_COL]
        ].join(
            mi_join, on=MOTIF_COL
        )
        motif_information = motif_information.dropna()

        def _renamer(s):
            s[MOTIF_OBJ_COL].motif_name = s[MOTIF_NAME_COL]

        motif_information.apply(_renamer, axis=1)
        motifs = motif_information[MOTIF_OBJ_COL].tolist()

        print(
            "Retained {n} / {m} motifs ({x} TFs)".format(
                n=len(motifs),
                m=_before_n,
                x=len(motif_information[MOTIF_NAME_COL].unique()),
            )
        )

    if regulator_constraint_list is not None:
        motifs = select_motifs(motifs, regulator_constraint_list)

    truncate_motifs(motifs, truncate_prob)

    return motifs, motif_information


def network_scan(
    motifs,
    motif_information,
    genes,
    genomic_fasta_file,
    constraint_bed_file=None,
    promoter_bed_file=None,
    scanner_type="fimo",
    num_cores=1,
    motif_ic=6,
    tandem=100,
    scanner_thresh="1e-4",
    debug=False,
    silent=False,
    save_locs=False,
):
    # Load and scan target chromatin peaks
    MotifScan.set_type(scanner_type)

    if debug:
        for chromosome, df in genes[GTF_CHROMOSOME].value_counts().iteritems():
            print("Chromosome {c}: {n} genes".format(c=chromosome, n=df))

    motif_peaks = MotifScan.scanner(motifs=motifs, num_workers=num_cores).scan(
        genomic_fasta_file,
        constraint_bed_file=constraint_bed_file,
        promoter_bed=promoter_bed_file,
        min_ic=motif_ic,
        threshold=scanner_thresh,
    )

    if debug:
        for chromosome, df in (
            motif_peaks[MotifScan.chromosome_col].value_counts().iteritems()
        ):
            print("Chromosome {c}: {n} motif hits".format(c=chromosome, n=df))

    if save_locs and motif_peaks is not None:
        print(f"Writing output file {save_locs}")
        motif_peaks.iloc[:, 0:-1].to_csv(save_locs, sep="\t", index=False)

    # PROCESS CHROMATIN PEAKS INTO NETWORK MATRIX #############################

    # Process into an information score matrix
    MotifScorer.set_information_criteria(
        min_binding_ic=motif_ic,
        max_dist=tandem
    )

    if motif_peaks is not None:
        raw_matrix, prior_data = summarize_target_per_regulator(
            genes,
            motif_peaks,
            motif_information,
            num_workers=num_cores,
            debug=debug,
            silent=silent,
        )
    else:
        raw_matrix = pd.DataFrame(
            0.0,
            index=genes[GTF_GENENAME],
            columns=motif_information[MOTIF_NAME_COL].unique().tolist(),
        )
        prior_data = None

    return raw_matrix, prior_data


def network_build(
    raw_matrix,
    prior_data,
    num_cores=1,
    output_prefix=None,
    debug=False,
    silent=False,
    save_locs_filtered=False,
):
    if output_prefix is not None:
        print(
            "Writing output file "
            f"{output_prefix + '_unfiltered_matrix.tsv.gz'}"
        )
        raw_matrix.to_csv(
            output_prefix + "_unfiltered_matrix.tsv.gz",
            sep="\t"
        )

    # Choose edges to keep
    prior_matrix = build_prior_from_motifs(
        raw_matrix, num_workers=num_cores, debug=debug, silent=silent
    )

    if output_prefix is not None:
        print(
            f"Writing output file {output_prefix + '_edge_matrix.tsv.gz'}"
        )
        (prior_matrix != 0).astype(int).to_csv(
            output_prefix + "_edge_matrix.tsv.gz", sep="\t"
        )

    if prior_data is not None:
        prior_matrix.index.name = PRIOR_GENE
        pm_melt = prior_matrix.reset_index().melt(
            id_vars=PRIOR_GENE, var_name=PRIOR_TF, value_name="Filter_Included"
        )
        prior_data = pd.merge(prior_data, pm_melt)

    if (
        save_locs_filtered and
        output_prefix is not None and
        prior_data is not None
    ):
        print(f"Writing output file {save_locs_filtered}")
        prior_data.to_csv(save_locs_filtered, sep="\t", index=False)

    return prior_matrix, raw_matrix, prior_data


def scan_and_build_by_tf(
    genomic_fasta_file,
    constraint_bed_file,
    genes,
    gene_locs,
    output_prefix,
    motif_information,
    debug=False,
    motif_ic=6,
    tandem=100,
    scanner_thresh="1e-4",
    save_locs=False,
    save_locs_filtered=False,
    num_cores=None,
    extract_genome=True,
    filter_motif_hits_for_gene_list=False,
    by_chromosome=True,
):
    if extract_genome:
        # EXTRACT GENOMIC SEQUENCES ONLY ONCE #################################
        extract_fasta = MotifScan.scanner.extract_genome(
            genomic_fasta_file,
            constraint_bed_file=constraint_bed_file,
            promoter_bed=gene_locs,
            debug=debug,
        )
    else:
        extract_fasta = genomic_fasta_file

    # BUILD PER-TF DATA FUNCTION ##############################################
    def network_scan_build_single_tf(tf_mi_df):
        tf_motifs = tf_mi_df[MOTIF_OBJ_COL].tolist()

        try:
            motif_peaks = MotifScan.scanner(
                motifs=tf_motifs,
                num_workers=1
            ).scan(
                None,
                extracted_genome=extract_fasta,
                min_ic=motif_ic,
                threshold=scanner_thresh,
            )
        except RuntimeError:
            return (None, None, None), None

        # Process into an information score matrix
        MotifScorer.set_information_criteria(
            min_binding_ic=motif_ic,
            max_dist=tandem
        )

        if motif_peaks is not None and filter_motif_hits_for_gene_list:
            motif_peaks = motif_peaks.loc[
                motif_peaks[MotifScan.chromosome_col].isin(
                    genes[GTF_GENENAME]
                ),
                :
            ]

        if motif_peaks is not None:
            ra_ma, pr_da = summarize_target_per_regulator(
                genes,
                motif_peaks,
                tf_mi_df,
                num_workers=1,
                debug=debug,
                silent=True,
                by_chromosome=by_chromosome,
            )

        else:
            ra_ma = pd.DataFrame(
                0.0,
                index=genes[GTF_GENENAME],
                columns=tf_mi_df[MOTIF_NAME_COL].unique().tolist(),
            )
            pr_da = None

        raw_loc = motif_peaks if save_locs else None
        net_results = network_build(
            ra_ma,
            pr_da,
            num_cores=1,
            output_prefix=None,
            debug=debug,
            silent=True
        )

        return net_results, raw_loc

    # MULTIPROCESS PER-TF #####################################################
    prior_matrix, raw_matrix, prior_data = [], [], []

    with pathos.multiprocessing.Pool(
        processes=num_cores,
        maxtasksperchild=10
    ) as pool:
        motif_information = [
            df
            for _, df in motif_information.groupby(MOTIF_NAME_COL)
        ]
        _is_first = True
        for i, res in enumerate(
            pool.imap_unordered(
                network_scan_build_single_tf,
                motif_information
            )
        ):
            if save_locs and res[1] is not None:
                res[1].iloc[:, 0:-1].to_csv(
                    save_locs,
                    sep="\t",
                    mode="w" if _is_first else "a",
                    header=_is_first,
                    index=False,
                )
                _is_first = False

            p_m, r_m, p_d = res[0]

            tf_name = p_m.columns[0] if p_m is not None else ""

            print(f"Processed TF {i}/{len(motif_information)} [{tf_name}]")
            prior_matrix.append(p_m)
            raw_matrix.append(r_m)
            prior_data.append(p_d)

    # CONCAT FINAL DATA #######################################################
    prior_matrix = pd.concat(prior_matrix, axis=1)
    raw_matrix = pd.concat(raw_matrix, axis=1)
    prior_data = pd.concat(prior_data, axis=0)

    if extract_genome and os.path.exists(extract_fasta):
        os.remove(extract_fasta)

    if output_prefix is not None:
        print(
            "Writing output file "
            f"{output_prefix + '_unfiltered_matrix.tsv.gz'}"
        )
        raw_matrix.to_csv(
            output_prefix + "_unfiltered_matrix.tsv.gz", sep="\t"
        )

    if output_prefix is not None:
        print(
            f"Writing output file {output_prefix + '_edge_matrix.tsv.gz'}"
        )
        (prior_matrix != 0).astype(int).to_csv(
            output_prefix + "_edge_matrix.tsv.gz", sep="\t"
        )

    if save_locs_filtered and prior_data is not None:
        prior_data.to_csv(save_locs_filtered, sep="\t", index=False)

    return prior_matrix, raw_matrix, prior_data


if __name__ == "__main__":
    main()
