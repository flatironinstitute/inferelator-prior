import pandas as pd
import pybedtools
import tempfile

from inferelator_prior.processor.gtf import (
    SEQ_START,
    SEQ_STOP,
    GTF_STRAND
)

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"

# Column names
BED_CHROMOSOME = 'chrom'
BED_COLS = [BED_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]

SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'
SEQ_SCORE = 'p-value'


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
    else:
        return pybedtools.BedTool(bed)


def intersect_bed(*beds, wa=False, wb=False):
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
        wb=wb
    )
