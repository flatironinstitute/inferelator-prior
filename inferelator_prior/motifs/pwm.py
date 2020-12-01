from inferelator_prior.motifs import Motif, MOTIF_COL

import pandas as pd
import pandas.errors as pde
import os

TF_NAME_COL = "TF_Name"
TF_STATUS_COL = "TF_Status"


def read(pwm_file_list, info_file, background=None, direct_only=False, pwm_has_idx=True, pwm_alphabet=None,
         transpose=False):

    info_df = pd.read_csv(info_file, sep="\t")
    info_df[MOTIF_COL] = info_df[MOTIF_COL].astype(str)

    motifs = []
    pwm_not_present = []
    pwm_malformed = []

    if not pwm_has_idx and pwm_alphabet is None:
        raise ValueError("pwm_alphabet is required if pwm_has_idx=False")

    for pwm_file in pwm_file_list:
        pwm_id = os.path.splitext(os.path.basename(pwm_file))[0]

        if pwm_id not in info_df[MOTIF_COL]:
            pwm_not_present.append(pwm_id)
            continue

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
            if pwm_has_idx:
                pwm = pd.read_csv(pwm_file, sep="\t", index_col=0)
            else:
                pwm = pd.read_csv(pwm_file, sep="\t", header=None)

            pwm = pwm.T if transpose else pwm
        except pde.ParserError:
            print("Parser error on file {f}".format(f=pwm_name))
            pwm_malformed.append(pwm_id)
            continue

        pwm_alphabet = pwm.columns.tolist() if pwm_has_idx else pwm_alphabet

        motif = Motif(pwm_id, pwm_name, pwm_alphabet, motif_background=background)
        motif.probability_matrix = pwm.values

        if min(pwm.values.shape) == 0:
            pwm_malformed.append(pwm_id)
            continue

        motifs.append(motif)

    if len(pwm_not_present) > 0:
        print("{pwm} PWM files not found in in {c} of {cf}".format(pwm=len(pwm_not_present), c=MOTIF_COL, cf=info_file))
    if len(pwm_malformed) > 0:
        print("{pwm} PWM files malformed and improperly parsed".format(pwm=len(pwm_malformed)))

    return motifs
