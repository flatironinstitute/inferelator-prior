from inferelator_prior.motifs.fimo import FIMO_MOTIF, FIMO_SCORE, FIMO_START, FIMO_STOP, FIMO_CHROMOSOME, FIMOScanner
from inferelator_prior.motifs.homer import HOMER_MOTIF, HOMER_SCORE, HOMER_START, HOMER_STOP, HOMER_CHROMOSOME, HOMERScanner
from inferelator_prior.motifs import meme
from inferelator_prior.motifs import homer_motif


class MotifScan(object):
    """
    This class handles keeping track of the info needed for each type of motif scanner
    """

    _motif_file_type = 'fimo'

    name_col = FIMO_MOTIF
    score_col = FIMO_SCORE
    chromosome_col = FIMO_CHROMOSOME
    start_col = FIMO_START
    stop_col = FIMO_STOP
    scanner = FIMOScanner

    @classmethod
    def set_type(cls, motif_type_str):
        # Set the scanner type
        if motif_type_str.lower() == 'fimo':
            cls.set_type_fimo()
        elif motif_type_str.lower() == 'homer':
            cls.set_type_homer()
        else:
            raise ValueError("motif_type must be fimo or homer")

    @classmethod
    def set_type_fimo(cls):
        cls.name_col = FIMO_MOTIF
        cls.score_col = FIMO_SCORE
        cls.chromosome_col = FIMO_CHROMOSOME
        cls.start_col = FIMO_START
        cls.stop_col = FIMO_STOP

        cls._motif_file_type = 'fimo'
        cls.scanner = FIMOScanner

    @classmethod
    def set_type_homer(cls):
        cls.name_col = HOMER_MOTIF
        cls.score_col = HOMER_SCORE
        cls.chromosome_col = HOMER_CHROMOSOME
        cls.start_col = HOMER_START
        cls.stop_col = HOMER_STOP

        cls._motif_file_type = 'homer'
        cls.scanner = HOMERScanner

    @classmethod
    def load_motif_file(cls, motif_file_name):
        if motif_file_name.lower().endswith(".meme"):
            return meme.read(motif_file_name)
        else:
            return homer_motif.read(motif_file_name)

