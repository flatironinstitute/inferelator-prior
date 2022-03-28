from inferelator_prior.motifs import Motif, MOTIF_COL

import pandas as pd
import pandas.errors as pde
import os

TF_NAME_COL = "TF_Name"
TF_STATUS_COL = "TF_Status"


def read(pwm_file_list, info_file, background=None, direct_only=False, pwm_has_idx=True, pwm_alphabet=None,
         transpose=False, tf_name_col=TF_NAME_COL, join_TF_names=True):

    info_df = pd.read_csv(info_file, sep="\t")
    info_df[MOTIF_COL] = info_df[MOTIF_COL].astype(str)

    motifs = []
    pwm_not_present = []
    pwm_malformed = []

    if not pwm_has_idx and pwm_alphabet is None:
        raise ValueError("pwm_alphabet is required if pwm_has_idx=False")

    for pwm_file in pwm_file_list:
        pwm_id = os.path.splitext(os.path.basename(pwm_file))[0]
        match_id = info_df[MOTIF_COL] == pwm_id

        if (match_id).sum() == 0:
            pwm_not_present.append(pwm_id)
            continue

        if direct_only:
            direct = info_df.loc[match_id, TF_STATUS_COL].str.contains("D")
            if not direct.any():
                continue
            else:
                pwm_names = info_df.loc[match_id & (info_df[TF_STATUS_COL] == "D"), tf_name_col]
        else:
            pwm_names = info_df.loc[match_id, tf_name_col]

        if join_TF_names:
            pwm_names = ["/".join(pwm_names)]

        try:
            if pwm_has_idx:
                pwm = pd.read_csv(pwm_file, sep="\t", index_col=0)
            else:
                pwm = pd.read_csv(pwm_file, sep="\t", header=None)

            pwm = pwm.T if transpose else pwm
        except pde.ParserError:
            print("Parser error on file {f}".format(f=pwm_id))
            pwm_malformed.append(pwm_id)
            continue

        pwm_alphabet = pwm.columns.tolist() if pwm_has_idx else pwm_alphabet

        for i, n in enumerate(pwm_names):
            if i > 0:
                pid = pwm_id + "_" + str(i)
            else:
                pid = pwm_id

            motif = Motif(pid, n, pwm_alphabet, motif_background=background)
            motif.probability_matrix = pwm.values

            if min(pwm.values.shape) == 0:

                if i == 0:
                    pwm_malformed.append(pwm_id)
                    
                continue

            motifs.append(motif)

    if len(pwm_not_present) > 0:
        print(f"{len(pwm_not_present)} PWM files not found in in {MOTIF_COL} of {info_file}")
    if len(pwm_malformed) > 0:
        print(f"{len(pwm_malformed)} PWM files malformed and improperly parsed")

    return motifs
