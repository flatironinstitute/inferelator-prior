import numpy as np
import pandas as pd
import warnings
import os
import shutil
import tempfile
import itertools
import pathos
from collections import Counter

from inferelator_prior.processor.bedtools import (extract_bed_sequence, intersect_bed, load_bed_to_bedtools,
                                                  BED_CHROMOSOME)
from inferelator_prior.processor.gtf import get_fasta_lengths, check_chromosomes_match

INFO_COL = "Information_Content"
ENTROPY_COL = "Shannon_Entropy"
OCC_COL = "Occurrence"
LEN_COL = "Length"
MOTIF_COL = "Motif_ID"
MOTIF_NAME_COL = "Motif_Name"
MOTIF_OBJ_COL = "Motif_Object"
MOTIF_CONSENSUS_COL = "Consensus"
MOTIF_ORIGINAL_NAME_COL = 'Motif_Name_Original'

SCAN_SCORE_COL = "Inferelator_Score"
SCORE_PER_BASE = "Per Base Array"

DEGEN_LOOKUP = {frozenset(("A", "T")): "W",
                frozenset(("A", "C")): "M",
                frozenset(("A", "G")): "R",
                frozenset(("C", "G")): "S",
                frozenset(("C", "T")): "Y",
                frozenset(("G", "T")): "K",
                frozenset("A"): "A",
                frozenset("T"): "T",
                frozenset("G"): "G",
                frozenset("C"): "C"}


class Motif:
    motif_id = None
    motif_name = None
    motif_url = None

    _motif_probs = None
    _motif_counts = None
    _motif_prob_array = None
    _motif_alphabet = None
    _motif_background = None
    _motif_species = None
    _motif_accession = None
    _alphabet_map = None
    _consensus_seq = None
    _consensus_seq_degen = None
    _info_matrix = None
    _homer_odds = None

    @property
    def alphabet(self):
        return self._motif_alphabet

    @alphabet.setter
    def alphabet(self, new_alphabet):
        if new_alphabet is not None:
            self._motif_alphabet = np.array(new_alphabet)
            self._alphabet_map = {ch.lower(): i for i, ch in enumerate(self._motif_alphabet)}

    @property
    def accession(self):
        return self._motif_accession

    @accession.setter
    def accession(self, new_accession):
        if new_accession is not None:
            self._motif_accession = new_accession

    @property
    def id(self):
        return self.motif_id

    @id.setter
    def id(self, new_id):
        if new_id is not None:
            self.motif_id = new_id

    @property
    def name(self):
        return self.motif_name

    @name.setter
    def name(self, new_name):
        if new_name is not None:
            self.motif_name = new_name

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
    def count_matrix(self):
        return np.array(self._motif_counts) if self._motif_counts is not None else None

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

        return np.sum(self.ic_matrix)

    @property
    def homer_odds(self):
        return self.threshold_ln_odds if self._homer_odds is None else self._homer_odds

    @homer_odds.setter
    def homer_odds(self, val):
        self._homer_odds = val

    @property
    def ic_matrix(self):
        if self.probability_matrix is None:
            return None

        if self._info_matrix is None or self._info_matrix.shape != self.probability_matrix.shape:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Calculate p log (p/background)
                self._info_matrix = np.divide(self.probability_matrix, self.background.reshape(1, -1))
                self._info_matrix = np.multiply(self.probability_matrix, np.log2(self._info_matrix))
                self._info_matrix[~np.isfinite(self._info_matrix)] = 0.
                self._info_matrix = np.maximum(self._info_matrix, 0.)

        return self._info_matrix

    @property
    def expected_occurrence_rate(self):
        return int(2 ** self.information_content)

    @property
    def consensus(self):
        if self._consensus_seq is None:
            self._consensus_seq = "".join(np.apply_along_axis(lambda x: self.alphabet[x.argmax()], axis=1,
                                                              arr=self.probability_matrix))
        return self._consensus_seq

    @property
    def consensus_degen(self):
        if self._consensus_seq_degen is None:
            def _csdegen(x):
                return DEGEN_LOOKUP[frozenset(self.alphabet[x >= 0.35])] if np.sum(x >= 0.35) > 0 else "N"

            self._consensus_seq_degen = "".join(np.apply_along_axis(_csdegen, axis=1, arr=self.probability_matrix))

        return self._consensus_seq_degen

    @property
    def max_ln_odds(self):
        max_ln_odd = np.log(np.amax(self.probability_matrix, axis=1) / 0.25)
        return np.sum(max_ln_odd)

    @property
    def threshold_ln_odds(self):
        second_prob = np.sort(self.probability_matrix, axis=1)[:, 2]
        return self.max_ln_odds - max((np.sum(np.log(second_prob[second_prob > 0.25] / 0.25)), 0.1 * self.max_ln_odds))

    @property
    def species(self):
        return self._motif_species

    @species.setter
    def species(self, new_species):
        is_list = isinstance(new_species, (list, tuple))

        if is_list and self._motif_species is None:
            self._motif_species = new_species
        elif is_list:
            self._motif_species.extend(new_species)
        elif self._motif_species is None:
            self._motif_species = [new_species]
        else:
            self._motif_species.append(new_species)

    def __len__(self):
        return self.probability_matrix.shape[0] if self.probability_matrix is not None else 0

    def __str__(self):
        return "{mid} {mname}: Width {el} IC {ic:.2f} bits".format(mid=self.motif_id,
                                                                   mname=self.motif_name,
                                                                   el=len(self),
                                                                   ic=self.information_content)

    def __eq__(self, other):
        try:
            return np.allclose(self.probability_matrix, other.probability_matrix) \
                   and (self.motif_id == other.motif_id) and (self.motif_name == other.motif_name)
        except AttributeError:
            pass

        try:
            return self.motif_name == other
        except TypeError:
            pass

        return False

    def __init__(self, motif_id=None, motif_name=None, motif_alphabet=None, motif_background=None):
        self.id = motif_id
        self.name = motif_name
        self.alphabet = np.array(motif_alphabet) if motif_alphabet is not None else None
        self._motif_background = motif_background
        self._motif_probs = []

    def add_prob_line(self, line):
        self._motif_probs.append(line)

    def add_count_line(self, line):
        if self._motif_counts is not None:
            self._motif_counts.append(line)
        else:
            self._motif_counts = [line]

    def score_match(self, match, disallow_homopolymer=True, homopolymer_one_off_len=6, score_zero_as_zero=None):

        if len(match) != len(self):
            msg = "Sequence length {l} not compatible with motif length {m}".format(l=len(match), m=len(self))
            raise ValueError(msg)

        # Score anything that's a homopolymer to 0 if the flag is set
        if disallow_homopolymer and sum([m == match[0] for m in match]) == len(match):
            return 0

        # Score anything that's one base from a homopolymer to 0 if the flag is set
        if disallow_homopolymer and (len(match) > homopolymer_one_off_len and
                                     sum([min((c, 2)) for c in Counter(match).values()]) < 4):
            return 0

        # Score anything with excessive nucleotides that have a p ~ 0.0 as 0
        if score_zero_as_zero is not None and sum(p < 0.001 for p in self._prob_match(match)) > score_zero_as_zero:
            return 0

        mse_ic = np.sum(np.square(np.subtract(self._info_match(self.consensus), self._info_match(match))))
        return max((np.sum(self._info_match(match)) - mse_ic, 0.))

    def truncate(self, threshold=0.35):
        threshold = np.max(self.probability_matrix, axis=1) > threshold
        keepers = (threshold.cumsum() > 0) & (threshold[::-1].cumsum()[::-1] > 0)
        self.probability_matrix = self.probability_matrix[keepers, :]
        self._motif_probs = list(itertools.compress(self._motif_probs, keepers))

    def _prob_match(self, match):
        return [self.probability_matrix[i, self._alphabet_map[ch.lower()]] for i, ch in enumerate(match)]

    def _info_match(self, match):
        return [self.ic_matrix[i, self._alphabet_map[ch.lower()]] for i, ch in enumerate(match)]

    def species_contains(self, match_str):
        if self.species is not None:
            match_str = match_str.lower()
            return any(match_str in s.lower() for s in self.species)
        else:
            return False

    def shuffle(self, rng=None, random_seed=42):
        """
        Shuffles per-base probabilities
        """

        if rng is not None:
            rng.shuffle(self.probability_matrix.T)
        else:
            np.random.default_rng(random_seed).shuffle(self.probability_matrix.T)


class MotifScanner:
    scanner_name = None

    def __init__(self, motif_file=None, motifs=None, num_workers=4):

        if (motif_file is None and motifs is None) or (motif_file is not None and motifs is not None):
            raise ValueError("One of meme_file or motifs must be passed")

        self.motif_file = motif_file
        self.motifs = motifs
        self.num_workers = num_workers

    def scan(self, genome_fasta_file=None, constraint_bed_file=None, promoter_bed=None, min_ic=None, threshold=None,
             valid_fasta_chromosomes=None, debug=False, extracted_genome=None):
        """
        """

        # Preprocess motifs into a list of temp chunk files
        motif_files = self._preprocess(min_ic=min_ic)
        # Unpack list to a dict for convenience
        self.motifs = {mot.motif_id: mot for mot in self.motifs}

        try:
            if extracted_genome is None:

                extracted_fasta_file = self.extract_genome(genome_fasta_file, constraint_bed_file, promoter_bed,
                                                           valid_fasta_chromosomes, debug)
                try:
                    motif_data = self._scan_extract(motif_files, extracted_fasta_file, threshold=threshold)
                    return self._postprocess(motif_data)
                finally:
                    try:
                        os.remove(extracted_fasta_file)
                    except FileNotFoundError:
                        pass

            else:
                motif_data = self._scan_extract(motif_files, extracted_genome, threshold=threshold)
                return self._postprocess(motif_data)

        finally:
            for file in motif_files:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass

    @staticmethod
    def extract_genome(genome_fasta_file, constraint_bed_file=None, promoter_bed=None,
                       valid_fasta_chromosomes=None, debug=False):

        if valid_fasta_chromosomes is None:
            _chr_lens = get_fasta_lengths(genome_fasta_file)
            valid_fasta_chromosomes = list(_chr_lens.keys())

        con_bed_file = load_bed_to_bedtools(constraint_bed_file) if constraint_bed_file is not None else None
        pro_bed_file = load_bed_to_bedtools(promoter_bed) if promoter_bed is not None else None

        if con_bed_file is not None and valid_fasta_chromosomes is not None:
            check_chromosomes_match(con_bed_file.to_dataframe(), valid_fasta_chromosomes,
                                    chromosome_column=BED_CHROMOSOME, file_name=constraint_bed_file)

            if debug:
                MotifScanner._print_bed_summary(con_bed_file, constraint_bed_file)

        if pro_bed_file is not None and valid_fasta_chromosomes is not None:
            check_chromosomes_match(pro_bed_file.to_dataframe(), valid_fasta_chromosomes,
                                    chromosome_column=BED_CHROMOSOME, file_name=pro_bed_file)

            if debug:
                MotifScanner._print_bed_summary(pro_bed_file, promoter_bed)

        if con_bed_file is not None and pro_bed_file is not None:
            bed_file = intersect_bed(load_bed_to_bedtools(constraint_bed_file), load_bed_to_bedtools(promoter_bed))
        elif con_bed_file is not None:
            bed_file = con_bed_file
        elif pro_bed_file is not None:
            bed_file = pro_bed_file
        else:
            extracted_fasta_file = tempfile.mkstemp(suffix=".fasta")[1]
            shutil.copy2(genome_fasta_file, extracted_fasta_file)
            return extracted_fasta_file

        extracted_fasta_file = extract_bed_sequence(bed_file, genome_fasta_file)
        return extracted_fasta_file

    def _scan_extract(self, motif_files, extracted_fasta_file, threshold=None, parse_genomic_coord=True):
        # If the number of workers is 1, run fimo directly
        if (self.num_workers == 1) or (len(motif_files) == 1):
            assert len(motif_files) == 1
            return self._get_motifs(extracted_fasta_file, motif_files[0], threshold=threshold,
                                    parse_genomic_coord=parse_genomic_coord)

        # Otherwise parallelize with a process pool (pathos because dill will do local functions)
        else:
            # Convenience local function
            n = len(motif_files)

            def _get_chunk_motifs(i, chunk_file):
                print("Launching {name} scanner [{i} / {n}]".format(name=self.scanner_name, i=i + 1, n=n))
                results = self._get_motifs(extracted_fasta_file, chunk_file, threshold=threshold,
                                           parse_genomic_coord=parse_genomic_coord)
                print("Scanning completed [{i} / {n}]".format(i=i + 1, n=n))
                return results

            with pathos.multiprocessing.Pool(self.num_workers) as pool:
                motif_data = [data for data in pool.starmap(_get_chunk_motifs, enumerate(motif_files))]
                motif_data = pd.concat(motif_data)

        return motif_data

    def _preprocess(self, min_ic=None):
        raise NotImplementedError

    def _postprocess(self, motif_peaks):
        raise NotImplementedError

    def _get_motifs(self, fasta_file, motif_file, threshold=None, parse_genomic_coord=True):
        raise NotImplementedError

    def _parse_output(self, output_handle):
        raise NotImplementedError

    @staticmethod
    def _print_bed_summary(bedtools_obj, bed_file_name):
        for chromosome, ct in bedtools_obj.to_dataframe()[BED_CHROMOSOME].value_counts().iteritems():
            print("BED File ({f}) parsing complete:".format(f=bed_file_name))
            print("\tChromosome {c}: {n} intervals found".format(c=chromosome, n=ct))


def motifs_to_dataframe(motifs):
    entropy = list(map(lambda x: x.shannon_entropy, motifs))
    occurrence = list(map(lambda x: x.expected_occurrence_rate, motifs))
    info = list(map(lambda x: x.information_content, motifs))
    ids = list(map(lambda x: x.motif_id, motifs))
    names = list(map(lambda x: x.motif_name, motifs))
    cons = list(map(lambda x: x.consensus_degen, motifs))

    df = pd.DataFrame(
        zip(ids, names, info, entropy, occurrence, list(map(lambda x: len(x), motifs)), motifs, cons),
        index=list(map(lambda x: x.motif_name, motifs)),
        columns=[MOTIF_COL, MOTIF_NAME_COL, INFO_COL, ENTROPY_COL, OCC_COL, LEN_COL, MOTIF_OBJ_COL, MOTIF_CONSENSUS_COL]
    )

    return df


def select_motifs(motifs, regulator_constraint_list):
    """
    Keep only motifs for TFs in a list. Case-insensitive.

    :param motifs: A list of motif objects
    :type motifs: list[Motif]
    :param regulator_constraint_list: A list of regulator names. Skip if None.
    :type regulator_constraint_list: list[str], None
    :return motifs: A list of motif objects
    :rtype: list[Motifs]
    """
    if regulator_constraint_list is None:
        return motifs

    if len(regulator_constraint_list) == 0:
        raise ValueError("No elements provided in regulator_constraint_list")

    _regulator_constraint_list = list(map(lambda x: x.upper(), regulator_constraint_list))

    _pre_len = len(motifs)
    motifs = [m for m in motifs if m.motif_name.upper() in _regulator_constraint_list]
    _retained_names = np.unique([m.motif_name for m in motifs])

    if len(motifs) == 0:
        _msg = "No overlap between motifs ({mo} ...) and constraint list ({li} ...)"
        _msg = _msg.format(mo=list(map(lambda x: x.motif_name, motifs[:min(3, len(motifs))])),
                           li=regulator_constraint_list[:min(3, len(regulator_constraint_list))])
        raise ValueError(_msg)

    print("{c} TFs Retained ({n} in constraint list, {t} / {al} motifs)".format(c=len(_retained_names),
                                                                                n=len(_regulator_constraint_list),
                                                                                t=len(motifs),
                                                                                al=_pre_len))
    return motifs


def truncate_motifs(motifs, truncate_value):
    """
    Remove flanking bases in motif until reaching a base that has a probability at least equal to truncate_value

    :param motifs: List of motif objects
    :type motifs: list[Motif]
    :param truncate_value: Required probability. None disables.
    :type truncate_value: numeric, None
    """

    if truncate_value is not None:
        [x.truncate(threshold=truncate_value) for x in motifs]


def shuffle_motifs(motifs, random_seed):
    """
    Shuffle probabilities for each base as a random control

    :param motifs: List of motif objects
    :type motifs: list[Motif]
    :param random_seed: Random generator seed
    :type random_seed: int
    """

    [m.shuffle(rng=np.random.default_rng(random_seed)) for m in motifs]


def fuzzy_merge_motifs(motif_dataframe, merge_col=MOTIF_NAME_COL, remove_dimers=False):

    motif_dataframe = motif_dataframe.copy()
    motif_dataframe[MOTIF_ORIGINAL_NAME_COL] = motif_dataframe[MOTIF_NAME_COL].copy()

    motif_dataframe['FUZZY_MATCH'] = motif_dataframe[merge_col].str.lower()

    # Drop any dashes (e.g ETS-1 ETS1 should merge)
    motif_dataframe['FUZZY_MATCH'] = motif_dataframe['FUZZY_MATCH'].str.replace("-", "", regex=False)

    # Drop anything in parens (e.g. Tal-1 (Scl) should just be Tal-1)
    motif_dataframe['FUZZY_MATCH'] = motif_dataframe['FUZZY_MATCH'].str.replace("\(.*\)", "", regex=True)

    # Polish up the ends to get rid of crap
    motif_dataframe['FUZZY_MATCH'] = motif_dataframe['FUZZY_MATCH'].str.strip(",-:\\/ \r\n")

    matched_data = []
    for motif_fuzz, fuzz_df in motif_dataframe.groupby('FUZZY_MATCH'):
        if len(fuzz_df) > 1:
            _nn = fuzz_df[MOTIF_NAME_COL].value_counts().index[0]
        else:
            _nn = fuzz_df[MOTIF_NAME_COL].iloc[0]

        def _renamer(obj):
            obj.motif_name = _nn

        fuzz_df['Old_Motif_Name'] = fuzz_df[MOTIF_NAME_COL]
        fuzz_df[MOTIF_OBJ_COL].apply(_renamer)
        fuzz_df[MOTIF_NAME_COL] = _nn

        matched_data.append(fuzz_df)

    matched_data = pd.concat(matched_data)

    if remove_dimers:
        matched_data = matched_data.loc[~matched_data[MOTIF_NAME_COL].str.contains(":"), :]
        matched_data = matched_data.loc[~matched_data[MOTIF_NAME_COL].str.contains(", "), :]
        matched_data = matched_data.loc[~matched_data[MOTIF_NAME_COL].str.contains(";"), :].copy()

    print("Fuzzy merging completed: {p} TFs reduced to {n} TFs".format(n=len(matched_data[MOTIF_NAME_COL].unique()),
                                                                       p=len(motif_dataframe[MOTIF_NAME_COL].unique())))
    return matched_data


def chunk_motifs(file_type, motifs, num_workers=4, min_ic=None):
    """
    Break a motif file up into chunks
    :param file_type: The meme or homer namespaces with a .read() and .write() function
    :type file_type: inferelator_prior.motifs parser
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
    num_workers = pathos.multiprocessing.cpu_count() if num_workers is None else num_workers

    if min_ic is not None:
        motifs = list(itertools.compress(motifs, [m.information_content >= min_ic for m in motifs]))

    if num_workers == 1:
        tf_h, file_name = tempfile.mkstemp(prefix='1', suffix=".mchunk", dir=temp_dir)
        with os.fdopen(tf_h, "w", 1) as tf_fh:
            file_type.write(tf_fh, motifs)
        return [file_name]

    num_workers = len(motifs) if num_workers > len(motifs) else num_workers
    chunk_index = np.repeat(np.arange(num_workers).reshape(1, -1), np.ceil(len(motifs) / num_workers), axis=0).flatten()
    chunk_index = chunk_index[0:len(motifs)]

    files = []

    for i in range(num_workers):
        tf_h, file_name = tempfile.mkstemp(prefix=str(i), suffix=".mchunk", dir=temp_dir)
        with os.fdopen(tf_h, "w", 1) as tf_fh:
            file_type.write(tf_fh, [m for m, b in zip(motifs, (chunk_index == i)) if b])
        files.append(file_name)

    return files
