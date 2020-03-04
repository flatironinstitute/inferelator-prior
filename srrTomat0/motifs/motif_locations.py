from srrTomat0.motifs.fimo import FIMO_MOTIF, FIMO_SCORE, FIMO_START, FIMO_STOP, FIMO_CHROMOSOME
from srrTomat0.processor.bedtools import merge_overlapping_peaks


class MotifLocationManager(object):
    """
    This class handles parsing of motif peak files
    """

    _motif_file_type = 'fimo'

    name_col = FIMO_MOTIF
    score_col = FIMO_SCORE
    chromosome_col = FIMO_CHROMOSOME
    start_col = FIMO_START
    stop_col = FIMO_STOP

    motif_data = None
    motif_names = None
    motif_scores = None

    @classmethod
    def get_motif_names(cls):
        if cls.motif_names is not None:
            return cls.motif_names
        elif cls.motif_data is not None:
            cls.motif_names = cls.motif_data[cls.name_col].unique().tolist()
            return cls.motif_names
        else:
            raise ValueError("Motif data has not been loaded")

    @classmethod
    def tf_score(cls, tf):
        if cls.motif_scores is not None:
            return cls.motif_scores[tf]
        elif cls.motif_data is not None:
            cls._calculate_scores()
            return cls.motif_scores[tf]
        else:
            raise ValueError("Motif data has not been loaded")

    @classmethod
    def _calculate_scores(cls):

        assert cls.name_col in cls.motif_data.columns
        assert cls.score_col in cls.motif_data.columns

        motif_scores = cls.motif_data.loc[:, [cls.name_col, cls.score_col]].groupby(cls.name_col)

        # Convert scores from a DataFrame to a dict, keyed by TF name, with a list of score values
        cls.motif_scores = {tf: scores[cls.score_col].tolist() for tf, scores in motif_scores}

    @classmethod
    def _assert_columns_exist(cls, df):
        assert cls.name_col in df.columns
        assert cls.score_col in df.columns
        assert cls.chromosome_col in df.columns
        assert cls.start_col in df.columns
        assert cls.stop_col in df.columns
