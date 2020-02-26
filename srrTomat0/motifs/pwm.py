from srrTomat0.motifs import Motif, MOTIF_COL, MOTIF_NAME_COL

import pandas as pd
import os


def read(pwm_file_list, info_file, background=None):

    info_data = pd.read_csv(info_file, sep="\t")
    motifs = []

    for pwm_file in pwm_file_list:
        pwm_id = os.path.splitext(os.path.basename(pwm_file))[0]

        pwm_name = "/".join(info_data.loc[info_data[MOTIF_COL] == pwm_id, MOTIF_NAME_COL])

        pwm = pd.read_csv(pwm_file, sep="\t", index_col=0)
        pwm_alphabet = pwm.columns.tolist()

        motif = Motif(pwm_id, pwm_name, pwm_alphabet, motif_background=background)
        motif.probability_matrix = pwm.values

        if min(pwm.values.shape) == 0:
            continue

        motifs.append(motif)

    return motifs
