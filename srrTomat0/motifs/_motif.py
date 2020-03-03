import numpy as np
import pandas as pd
import warnings
import os
import tempfile
import math
import itertools
import pathos

from srrTomat0.processor.bedtools import extract_bed_sequence

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
        if self._motif_prob_array is None and len(self._motif_probs) == 0:
            return None
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
        if self.probability_matrix is None:
            return 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Calculate p log (p/background)
            info = np.divide(self.probability_matrix, self.background[:, np.newaxis])
            info = np.multiply(self.probability_matrix, np.log2(info))
            info[~np.isfinite(info)] = 0

        return np.sum(info)

    @property
    def expected_occurrence_rate(self):

        # Background correct probabilities
        occ_rate = np.divide(self.probability_matrix, self.background)
        occ_rate = np.max(occ_rate, axis=1)

        return int(np.prod(occ_rate))

    def __len__(self):
        return self.probability_matrix.shape[0] if self.probability_matrix is not None else 0

    def __str__(self):
        return "{mid} {mname}: Width {el} IC {ic:.2f} bits".format(mid=self.motif_id,
                                                                   mname=self.motif_name,
                                                                   el=len(self),
                                                                   ic=self.information_content)

    @property
    def consensus(self):
        con_seq = np.apply_along_axis(lambda x: self.alphabet[x.argmax()], axis=1,
                                      arr=self.probability_matrix)
        return "".join(con_seq)

    @property
    def max_ln_odds(self):
        max_ln_odd = np.log(np.amax(self.probability_matrix, axis=1) / 0.25)
        return np.sum(max_ln_odd)

    @property
    def threshold_ln_odds(self):
        second_prob = np.sort(self.probability_matrix, axis=1)[:, 2]
        return self.max_ln_odds - max((np.sum(np.log(second_prob[second_prob > 0.25] / 0.25)), 0.1 * self.max_ln_odds))

    def __init__(self, motif_id, motif_name, motif_alphabet, motif_background=None):
        self.motif_id = motif_id
        self.motif_name = motif_name
        self._motif_alphabet = motif_alphabet
        self._motif_background = motif_background
        self._motif_probs = []

    def add_prob_line(self, line):
        self._motif_probs.append(line)


class MotifScanner:

    def __init__(self, motif_file=None, motifs=None, num_workers=4):

        if (motif_file is None and motifs is None) or (motif_file is not None and motifs is not None):
            raise ValueError("One of meme_file or motifs must be passed")

        self.motif_file = motif_file
        self.motifs = motifs
        self.num_workers = num_workers

    def scan(self, atac_bed_file, genome_fasta_file, min_ic=None):
        """
        """

        # Extract interesting sequences to a temp fasta file
        extracted_fasta_file = extract_bed_sequence(atac_bed_file, genome_fasta_file)
        print(extracted_fasta_file)
        # Preprocess motifs into a list of temp chunk files
        meme_files = self._preprocess(min_ic=min_ic)

        try:

            # If the number of workers is 1, run fimo directly
            if self.num_workers == 1:
                assert len(meme_files) == 1
                return self._get_motifs(extracted_fasta_file, meme_files[0])

            # Otherwise parallelize with a process pool (pathos because dill will do local functions)
            else:
                # Convenience local function
                def _get_chunk_motifs(chunk_file):
                    return self._get_motifs(extracted_fasta_file, chunk_file)

                with pathos.multiprocessing.Pool(self.num_workers) as pool:
                    motif_data = [data for data in pool.imap(_get_chunk_motifs, meme_files)]
                    motif_data = pd.concat(motif_data)

            return motif_data

        # Clean up the temporary files
        finally:
            os.remove(extracted_fasta_file)

            for file in meme_files:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass

    def _preprocess(self, min_ic=None):
        raise NotImplementedError

    def _get_motifs(self, fasta_file, motif_file):
        raise NotImplementedError

    def _parse_output(self, output_handle):
        raise NotImplementedError


def motifs_to_dataframe(motifs):

    entropy = list(map(lambda x: x.shannon_entropy, motifs))
    occurrence = list(map(lambda x: x.expected_occurrence_rate, motifs))
    info = list(map(lambda x: x.information_content, motifs))

    df = pd.DataFrame(
        [list(map(lambda x: x.motif_id, motifs)), info, entropy, occurrence, list(map(lambda x: len(x), motifs))],
        columns=list(map(lambda x: x.motif_name, motifs)),
        index=[MOTIF_COL, INFO_COL, ENTROPY_COL, OCC_COL, LEN_COL]).T

    return df


def chunk_motifs(file_type, motif_file=None, motifs=None, num_workers=4, min_ic=None):
    """
    Break a motif file up into chunks
    :param file_type: The meme or homer namespaces with a .read() and .write() function
    :type file_type: srrTomat0.motifs parser
    :param motif_file: File name; pass either meme_file or motifs
    :type motif_file: str, None
    :param motifs: Motif object list; pass either meme_file or motifs
    :type motifs: list(Motif), None
    :param num_workers: number of chunks to make
    :type num_workers: int
    :param min_ic: set an information content minimum on motifs to include if this is not None
    :type min_ic: float
    :return: List of chunked motif files
    :rtype: list
    """

    temp_dir = tempfile.gettempdir()

    if (motif_file is None and motifs is None) or (motif_file is not None and motifs is not None):
        raise ValueError("One of motif_file or motifs must be passed")
    if motif_file is not None:
        motifs = file_type.read(motif_file)

    if min_ic is not None:
        motifs = list(itertools.compress(motifs, [m.information_content >= min_ic for m in motifs]))

    if num_workers == 1:
        file_name = os.path.join(temp_dir, "chunk1.mchunk")
        file_type.write(file_name, motifs)
        return [file_name]

    chunk_size = math.ceil(len(motifs) / num_workers)

    files = []

    for i in range(num_workers):
        file_name = os.path.join(temp_dir, "chunk" + str(i) + ".mchunk")
        file_type.write(file_name, motifs[i * chunk_size:min((i+1) * chunk_size, len(motifs))])
        files.append(file_name)

    return files

