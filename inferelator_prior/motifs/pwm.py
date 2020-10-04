from inferelator_prior.motifs import Motif, MOTIF_COL

import pandas as pd
import pandas.errors as pde
import os

TF_NAME_COL = "TF_Name"
TF_STATUS_COL = "TF_Status"


def read(pwm_file_list, info_file, background=None, direct_only=False):

    info_df = pd.read_csv(info_file, sep="\t")
    motifs = []

    for pwm_file in pwm_file_list:
        pwm_id = os.path.splitext(os.path.basename(pwm_file))[0]

        if direct_only:
            direct = info_df.loc[info_df[MOTIF_COL] == pwm_id, TF_STATUS_COL].str.contains("D")
            if not direct.any():
                continue
            else:
                pwm_names = info_df.loc[(info_df[MOTIF_COL] == pwm_id) & (info_df[TF_STATUS_COL] == "D"), TF_NAME_COL]
        else:
            pwm_names = info_df.loc[info_df[MOTIF_COL] == pwm_id, TF_NAME_COL]

        pwm_name = "/".join(pwm_names)

        try:
            pwm = pd.read_csv(pwm_file, sep="\t", index_col=0)
        except pde.ParserError:
            print("Parser error on file {f}".format(f=pwm_name))
            continue

        pwm_alphabet = pwm.columns.tolist()

        motif = Motif(pwm_id, pwm_name, pwm_alphabet, motif_background=background)
        motif.probability_matrix = pwm.values

        if min(pwm.values.shape) == 0:
            continue

        motifs.append(motif)

    return motifs
