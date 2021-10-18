import pandas as pd
import pybedtools
import tempfile

from inferelator_prior.processor.gtf import SEQ_START, SEQ_STOP, GTF_STRAND

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"

# Column names
BED_CHROMOSOME = 'chrom'

SEQ_COUNTS = 'count'
SEQ_BIN = 'bin'
SEQ_SCORE = 'p-value'


def load_bed_to_dataframe(bed_file_path, **kwargs):
    """
    :param bed_file_path: str
    :return: pd.DataFrame
    """

    _colnames = [BED_CHROMOSOME, SEQ_START, SEQ_STOP, GTF_STRAND]

    # Check and see if there are headers
    first_2_cols = pd.read_csv(bed_file_path, sep="\t", index_col=None, **kwargs, nrows=2)

    # Use the file headers if they exist    
    if any([x in first_2_cols.columns for x in _colnames]):

        # Warn if they're weird
        if not first_2_cols.columns.tolist()[0:3] == _colnames[0:3]:
            print("Nonstandard BED heade for file {f}: {r}".format(f=bed_file_path, r=first_2_cols.columns.tolist()))
        
        return pd.read_csv(bed_file_path, sep="\t", index_col=None, **kwargs)

    # If not, use the standard headers
    if first_2_cols.shape[1] > 4:
        _colnames = _colnames + list(map(lambda x: str(x), range(0, first_2_cols.shape[1] - 4)))

    return pd.read_csv(bed_file_path, sep="\t", index_col=None, names=_colnames[0:first_2_cols.shape[1]], **kwargs)


def extract_bed_sequence(bed_file, genome_fasta, output_path=None):
    output_path = tempfile.gettempdir() if output_path is None else output_path
    output_fh, output_file = tempfile.mkstemp(prefix="genome", suffix=BEDTOOLS_EXTRACT_SUFFIX, dir=output_path)

    if not isinstance(bed_file, pybedtools.BedTool):
        bed_file = pybedtools.BedTool(bed_file)

    try:
        bed_file.sequence(fi=genome_fasta, fo=output_file)
    except pybedtools.helpers.BEDToolsError as pbe:
        print(pbe.msg)

    return output_file


def load_bed_to_bedtools(bed):
    if bed is None:
        return None
    elif isinstance(bed, pd.DataFrame):
        return pybedtools.BedTool.from_dataframe(bed)
    else:
        return pybedtools.BedTool(bed)


def intersect_bed(*beds, wa=False, wb=False):

    if len(beds) == 1:
        return beds[0]

    beds = [b.sort() for b in beds]
    return beds[0].intersect(beds[1:], sorted=True, wa=wa, wb=wb)
