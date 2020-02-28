from srrTomat0.motifs.fimo import parse_fimo_output, FIMO_MOTIF, FIMO_SCORE, FIMO_START, FIMO_STOP, FIMO_CHROMOSOME
from srrTomat0.processor.bedtools import merge_overlapping_peaks


class MotifLocationManager(object):
    """
    This class handles parsing of motif peak files
    """

    _motif_file_type = 'fimo'
    _motif_parser = parse_fimo_output

    name_col = FIMO_MOTIF
    score_col = FIMO_SCORE
    chromosome_col = FIMO_CHROMOSOME
    start_col = FIMO_START
    stop_col = FIMO_STOP

    motif_data = None
    motif_names = None
    motif_scores = None

    @classmethod
    def get_motifs(cls, motif_bed_file):

        # Load and merge motif peaks
        print("Loading TF Motif Peaks from file ({f}) [{fmt} format]".format(f=motif_bed_file,
                                                                             fmt=cls._motif_file_type))
        motifs = cls._motif_parser(motif_bed_file)

        print("\t{n} peaks loaded".format(n=motifs.shape[0]))

        cls._assert_columns_exist(motifs)
        motifs = merge_overlapping_peaks(motifs,
                                         feature_group_column=cls.name_col,
                                         score_columns=[(cls.score_col, 'max')],
                                         start_column=cls.start_col,
                                         end_column=cls.stop_col,
                                         chromosome_column=cls.chromosome_col)

        print("\t{n} peaks remain after merge".format(n=motifs.shape[0]))

        cls.motif_data = motifs
        return motifs

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
    def set_motif_file_type(cls, motif_file_type):
        if motif_file_type == 'fimo':
            cls._set_type_fimo()
        else:
            raise ValueError("motif_type value {v} not supported".format(v=motif_file_type))

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
    def _set_type_fimo(cls):
        cls._motif_file_type = 'fimo'
        cls._motif_parser = parse_fimo_output

        cls.name_col = FIMO_MOTIF
        cls.score_col = FIMO_SCORE
        cls.chromosome_col = FIMO_CHROMOSOME
        cls.start_col = FIMO_START
        cls.stop_col = FIMO_STOP

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
