import pandas as pd
import numpy as np
import pybedtools
import tempfile
import gzip

from inferelator_prior.processor.gtf import (
    load_gtf_to_dataframe,
    open_window,
    check_chromosomes_match,
    GTF_CHROMOSOME,
    SEQ_START,
    SEQ_STOP,
    GTF_STRAND,
    GTF_GENENAME,
    SEQ_TSS
)

from inferelator.utils import is_string

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"

# Column names
BED_CHROMOSOME = 'chrom'
BED_COLS = [BED_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]

SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'
SEQ_SCORE = 'p-value'

GENE_COL = "gene"
PEAK_COL = "peak"
DIST_COL = "distance"


def load_bed_to_dataframe(
    bed_file_path,
    **kwargs
):
    """
    Load a BED file into a dataframe

    :param bed_file_path: Bed file path
    :type bed_file_path: str
    :return: BED records as a dataframe
    :rtype: pd.DataFrame
    """

    # Check and see if there are headers
    first_2_cols = pd.read_csv(
        bed_file_path,
        sep="\t",
        index_col=None,
        **kwargs,
        nrows=2
    )

    # Use the file headers if they exist
    if any([x in first_2_cols.columns for x in BED_COLS]):

        # Warn if they're weird
        if not first_2_cols.columns.tolist()[0:3] == BED_COLS[0:3]:
            print(
                f"Nonstandard BED header for file {bed_file_path}: "
                f"{first_2_cols.columns.tolist()}"
            )

        return pd.read_csv(
            bed_file_path,
            sep="\t",
            index_col=None,
            **kwargs
        )

    # If not, use the standard headers
    # Padding if needed for extra columns
    if first_2_cols.shape[1] > 4:
        _colnames = BED_COLS + list(
            map(
                lambda x: str(x),
                range(0, first_2_cols.shape[1] - 4)
            )
        )
    else:
        _colnames = BED_COLS[0:first_2_cols.shape[1]]

    return pd.read_csv(
        bed_file_path,
        sep="\t",
        index_col=None,
        names=_colnames,
        **kwargs
    )


def extract_bed_sequence(
    bed_file,
    genome_fasta,
    output_path=None
):
    """
    Extract BED sequences from a genomic FASTA to a
    separate FASTA file

    :param bed_file: BED file path or pyBEDtools object
    :type bed_file: str, pybedtools.BedTool
    :param genome_fasta: Genomic FASTA file path
    :type genome_fasta: str
    :param output_path: Path for the extracted file,
        will use system $TEMPDIR if None,
        defaults to None
    :type output_path: str, optional
    :return: Path to the extracted temporary FASTA file
    :rtype: str
    """

    if output_path is None:
        output_path = tempfile.gettempdir()

    output_fh, output_file = tempfile.mkstemp(
        prefix="genome",
        suffix=BEDTOOLS_EXTRACT_SUFFIX,
        dir=output_path
    )

    if not isinstance(bed_file, pybedtools.BedTool):
        bed_file = load_bed_to_bedtools(bed_file)

    try:
        bed_file.sequence(fi=genome_fasta, fo=output_file)

    except pybedtools.helpers.BEDToolsError as pbe:
        print(pbe.msg)

    return output_file


def load_bed_to_bedtools(bed):
    """
    Create a pybedtools BedTool object from a file
    or a pandas dataframe

    :param bed: DataFrame object or Path to file
    :type bed: pd.DataFrame, str
    :return: BedTool object
    :rtype: pybedtools.BedTool
    """

    if bed is None:
        return None

    elif isinstance(bed, pd.DataFrame):
        return pybedtools.BedTool.from_dataframe(bed)

    elif is_string(bed) and bed.endswith(".gz"):
        return pybedtools.BedTool(gzip.open(bed))

    else:
        return pybedtools.BedTool(bed)


def intersect_bed(
    *beds,
    wa=False,
    wb=False,
    **kwargs
):
    """
    Intersect 2 or more BED files

    :param *beds: Arbitrary number of BedTool objects
    :type beds: pybedtools.BedTool
    :param wa: Set the bedtools intersect -wa flag,
        defaults to False
    :type wa: bool, optional
    :param wb: Set the bedtools intersect -wb flag,
        defaults to False
    :type wb: bool, optional
    :return: Intersected BedTools object
    :rtype: pybedtools.BedTool
    """

    if len(beds) == 1:
        return beds[0]

    # Sort each bedtools object
    beds = [b.sort() for b in beds]

    return beds[0].intersect(
        beds[1:],
        sorted=True,
        wa=wa,
        wb=wb,
        **kwargs
    )


def link_bed_to_genes(
    bed_file,
    gene_annotation_file,
    out_file,
    use_tss=True,
    window_size=1000,
    dprint=print,
    non_gene_key="Intergenic",
    out_header=False,
    add_distance=False,
    check_chromosomes=True
):
    """
    Link a BED file (of arbitraty origin) to a set of genes from a GTF file
    based on proximity

    :param bed_file: Path to the BED file
    :type bed_file: str
    :param gene_annotation_file: Path to the genome annotation file (GTF)
    :type gene_annotation_file: str
    :param out_file: Path to the output file or open file handle
    :type out_file: str, handle
    :param use_tss: Base gene proximity on the TSS, not the gene body;
        defaults to True
    :type use_tss: bool, optional
    :param window_size: Window size (N, M) for proximity, where N is upstream
        of the gene and M is downstream.
        If given as an integer K, interpreted as (K, K); defaults to 1000
    :type window_size: int, tuple, optional
    :param dprint: Debug message function (can be overridden to silence),
        defaults to print
    :type dprint: callable, optional
    :param non_gene_key: Name for BED peaks that aren't in the genome feature
        windows. Set to None to drop peaks that aren't in the genome feature
        windows; defaults to "Intergenic"
    :type non_gene_key: str, optional
    :return: Number of peaks before mapping, number of peaks after mapping,
        dataframe of peaks
    :rtype: int, int, pd.DataFrame
    """

    if isinstance(gene_annotation_file, str):
        dprint(f"Loading genes from file ({gene_annotation_file})")

        # Load genes and open a window
        genes = load_gtf_to_dataframe(gene_annotation_file)
        dprint(f"{genes.shape[0]} genes loaded")

    elif isinstance(gene_annotation_file, pd.DataFrame):
        genes = gene_annotation_file.copy()

    dprint(
        f"Promoter regions defined with window {window_size} "
        f"around {'TSS' if use_tss else 'gene'}"
    )

    if window_size is not None:
        genes_window = open_window(
            genes,
            window_size=window_size,
            use_tss=use_tss,
            include_entire_gene_body=True,
            constrain_to_intergenic=True
        )
    else:
        genes_window = genes

    # Create a fake bed file with the gene promoter
    _all_cols = [GTF_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND, GTF_GENENAME]
    _int_cols = [SEQ_START, SEQ_STOP]

    genes_window = genes_window[_all_cols].copy()
    genes_window[_int_cols] = genes_window[_int_cols].astype(int)
    genes_window = genes_window.rename(
        {GTF_CHROMOSOME: BED_CHROMOSOME},
        axis=1
    ).sort_values(
        by=[BED_CHROMOSOME, SEQ_START]
    )

    gene_bed = load_bed_to_bedtools(genes_window)

    # Load BED-type file to a dataframe
    # Explicitly cast chromosomes into strings
    # (edge condition when chromosomes are just 1, 2, 3, Mt, etc)

    if isinstance(bed_file, str):
        dprint(f"Loading BED from file ({bed_file})")

        # Load genes and open a window
        bed_df = load_bed_to_dataframe(bed_file)
        dprint(f"{bed_df.shape[0]} BED annotations loaded")

    elif isinstance(bed_file, pd.DataFrame):
        bed_df = bed_file.copy()

    bed_df[BED_CHROMOSOME] = bed_df[BED_CHROMOSOME].astype(str)

    if check_chromosomes:
        check_chromosomes_match(
            genes_window,
            bed_df[BED_CHROMOSOME].unique().tolist(),
            chromosome_column=BED_CHROMOSOME
        )

    bed_locs = load_bed_to_bedtools(bed_df)

    try:
        ia = intersect_bed(
            gene_bed,
            bed_locs,
            wb=True
        )

        _new_cols = [
            'a_chrom',
            'a_start',
            'a_end',
            GTF_STRAND,
            GENE_COL,
            BED_CHROMOSOME,
            SEQ_START,
            SEQ_STOP
        ]

        # Check to see if there are extra mystery fields
        # And give them names
        if ia.field_count() > 8:
            _extra_cols = list(
                map(
                    str,
                    range(8, ia.field_count())
                )
            )
        else:
            _extra_cols = []

        # Create a dataframe and select the necessary
        # columns to make a BED file
        ia = ia.to_dataframe(
            names=_new_cols + _extra_cols
        )

        ia = ia[[
            BED_CHROMOSOME,
            SEQ_START,
            SEQ_STOP,
            GTF_STRAND,
            GENE_COL
        ]].copy()

    # Print some of the file structures
    # because there's a structure problem if intersect fails
    except:
        print("Gene BED file:")
        print(gene_bed.to_dataframe().head())
        print("Target BED dataframe:")
        print(bed_df.head())
        print("Target BED file:")
        print(bed_locs.to_dataframe().head())
        raise

    # Add an intergenic key if set
    # otherwise peaks that don't overlap will be dropped
    if non_gene_key is not None:
        ia = ia.merge(
            bed_df,
            how="outer",
            on=[BED_CHROMOSOME, SEQ_START, SEQ_STOP],
            suffixes=(None, "_bed")
        )

        ia[GENE_COL] = ia[GENE_COL].fillna(non_gene_key)
        ia[GTF_STRAND] = ia[GTF_STRAND].fillna(".")

    # Make unique peak IDs based on gene
    ia[PEAK_COL] = ia[GENE_COL].groupby(
        ia[GENE_COL]
    ).transform(
        lambda x: pd.Series(
            map(lambda y: "_" + str(y), range(len(x))),
            index=x.index
        )
    )

    ia[PEAK_COL] = ia[GENE_COL].str.cat(ia[PEAK_COL])

    peaks = ia[PEAK_COL].copy()
    ia.drop(PEAK_COL, inplace=True, axis=1)
    ia.insert(5, PEAK_COL, peaks)

    if add_distance:
        ia = ia.merge(
            genes.rename(
                {GTF_GENENAME: GENE_COL},
                axis=1
            )[[GENE_COL, SEQ_TSS]],
            on=GENE_COL
        )

        dists = ia[[SEQ_START, SEQ_STOP]].values - ia[[SEQ_TSS]].values

        ia[DIST_COL] = np.abs(dists).min(axis=1)

        _overlaps = np.prod(np.sign(dists), axis=1) == -1
        ia.loc[_overlaps, DIST_COL] = 0

    # Sort for output
    ia = ia.sort_values(
        by=[BED_CHROMOSOME, SEQ_START]
    ).reset_index(
        drop=True
    )

    if out_file is not None:
        ia.to_csv(
            out_file,
            sep="\t",
            index=False,
            header=out_header
        )

    return bed_locs.count(), len(ia), ia
