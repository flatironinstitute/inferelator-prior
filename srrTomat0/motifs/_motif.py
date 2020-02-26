import numpy as np
import pandas as pd
import warnings

INFO_COL = "Information Content"
ENTROPY_COL = "Shannon Entropy"
OCC_COL = "Occurrence"
LEN_COL = "Length"
MOTIF_COL = "Motif_ID"
MOTIF_NAME_COL = "Motif_Name"


class Motif:
    motif_id = None
    motif_name = None
    motif_url = None

    _motif_probs = None
    _motif_prob_array = None
    _motif_alphabet = None
    _motif_background = None

    @property
    def alphabet(self):
        return self._motif_alphabet

    @property
    def alphabet_len(self):
        return len(self._motif_alphabet)

    @property
    def background(self):
        if self._motif_background is None:
            self._motif_background = np.array([[1 / self.alphabet_len] * self.alphabet_len])
        return self._motif_background

    @property
    def probability_matrix(self):
        if self._motif_prob_array is None or self._motif_prob_array.shape[0] < len(self._motif_probs):
            self._motif_prob_array = np.array(self._motif_probs)
        return self._motif_prob_array

    @probability_matrix.setter
    def probability_matrix(self, matrix):
        self._motif_prob_array = matrix

    @property
    def shannon_entropy(self):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Calculate -1 * p log p and set to 0 where p is already 0
            entropy = np.multiply(self.probability_matrix, np.log2(self.probability_matrix))
            entropy[~np.isfinite(entropy)] = 0
            entropy *= -1

        return np.sum(entropy)

    @property
    def information_content(self):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            try:
                # Calculate p log (p/background)
                info = np.divide(self.probability_matrix, self.background[:, np.newaxis])
                info = np.multiply(self.probability_matrix, np.log2(info))
                info[~np.isfinite(info)] = 0
            except:
                print(info)
                raise

        return np.sum(info)

    @property
    def expected_occurrence_rate(self):

        # Background correct probabilities
        occ_rate = np.divide(self.probability_matrix, self.background)
        occ_rate = np.max(occ_rate, axis=1)

        return int(np.prod(occ_rate))

    def __len__(self):
        return self.probability_matrix.shape[0]

    def __str__(self):
        return "{mid} {mname}: Width {el} IC {ic:.2f} bits".format(mid=self.motif_id,
                                                                   mname=self.motif_name,
                                                                   el=len(self),
                                                                   ic=self.information_content)

    def __init__(self, motif_id, motif_name, motif_alphabet, motif_background=None):
        self.motif_id = motif_id
        self.motif_name = motif_name
        self._motif_alphabet = motif_alphabet
        self._motif_background = motif_background
        self._motif_probs = []

    def add_prob_line(self, line):
        self._motif_probs.append(line)


def motifs_to_dataframe(motifs):

    entropy = list(map(lambda x: x.shannon_entropy, motifs))
    occurrence = list(map(lambda x: x.expected_occurrence_rate, motifs))
    info = list(map(lambda x: x.information_content, motifs))

    df = pd.DataFrame(
        [list(map(lambda x: x.motif_id, motifs)), info, entropy, occurrence, list(map(lambda x: len(x), motifs))],
        columns=list(map(lambda x: x.motif_name, motifs)),
        index=[MOTIF_COL, INFO_COL, ENTROPY_COL, OCC_COL, LEN_COL]).T

    return df

