import unittest
import os
import io
import pandas as pd
import numpy as np
import numpy.testing as npt
import copy

from inferelator_prior.motifs import meme, homer_motif, transfac, shuffle_motifs

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")

MEME_FILE_NAME = "test.meme"
TRANSFAC_FILE_NAME = "test.transfac.dat"
MOTIF_FILE_NAME = "test.motif"
PWM_FILE_NAME = "M00799_2.00.txt"
ECORI_FILE_NAME = "test_ecori.meme"

TEST_MOTIF_MATRIX = """\
0.248650039776609	0.26139859992769	0.241301320519092	0.248650039776609
0.392226269785661	0.219606847798542	0.0702344472606129	0.317932435155184
0.0257615986027584	0.0113462234969035	0.942829024734492	0.0200631531658465
0.931183694119653	0.0181098604626899	0.00917476304082202	0.0415316823768348
0.0209101275685474	0.0204960793014344	0.00500398009361691	0.953589813036401
0.652610298711976	0.0667517267238459	0.0282467996828702	0.252391174881308
0.541666052707409	0.110459581183674	0.158324966820951	0.189549399287965
0.196060485729292	0.286696226860535	0.350171562229744	0.167071725180429
0.286129811839987	0.225077681884186	0.309960724838339	0.178831781437488
"""

PWM = pd.read_csv(io.StringIO(TEST_MOTIF_MATRIX), sep="\t", index_col=None, header=None).values


class TestMotifParsers(unittest.TestCase):

    def test_meme_loader(self):

        meme_file_name = os.path.join(artifact_path, MEME_FILE_NAME)
        motifs = meme.read(meme_file_name)

        with open(meme_file_name) as meme_fh:
            motifs2 = meme.read(meme_fh)

        self.assertEqual(len(motifs), 1)
        npt.assert_array_almost_equal(motifs[0].probability_matrix, PWM, 4)
        npt.assert_array_almost_equal(motifs2[0].probability_matrix, PWM, 4)

        self.assertListEqual(motifs[0].alphabet.tolist(), list("ACGT"))
        self.assertEqual(motifs[0].alphabet_len, 4)
        self.assertAlmostEqual(motifs[0].information_content, 7.6857, 3)

    def test_homer_motif_loader(self):

        motif_file_name = os.path.join(artifact_path, MOTIF_FILE_NAME)
        motifs = homer_motif.read(motif_file_name)

        with open(motif_file_name) as meme_fh:
            motifs2 = homer_motif.read(meme_fh)

        self.assertEqual(len(motifs), 1)
        npt.assert_array_almost_equal(motifs[0].probability_matrix, PWM, 4)
        npt.assert_array_almost_equal(motifs2[0].probability_matrix, PWM, 4)

        self.assertListEqual(motifs[0].alphabet.tolist(), list("ACGT"))
        self.assertEqual(motifs[0].alphabet_len, 4)
        self.assertAlmostEqual(motifs[0].information_content, 7.6857, 3)

    def test_pwm_loader(self):

        motif_file_name = os.path.join(artifact_path, MOTIF_FILE_NAME)
        motifs = homer_motif.read(motif_file_name)

        with open(motif_file_name) as meme_fh:
            motifs2 = homer_motif.read(meme_fh)

        self.assertEqual(len(motifs), 1)
        npt.assert_array_almost_equal(motifs[0].probability_matrix, PWM, 4)
        npt.assert_array_almost_equal(motifs2[0].probability_matrix, PWM, 4)

        self.assertListEqual(motifs[0].alphabet.tolist(), list("ACGT"))
        self.assertEqual(motifs[0].alphabet_len, 4)
        self.assertAlmostEqual(motifs[0].information_content, 7.6857, 3)

    def test_transfac_loader(self):

        motif_file_name = os.path.join(artifact_path, TRANSFAC_FILE_NAME)
        motifs = transfac.read(motif_file_name)

        with open(motif_file_name) as fh:
            motifs2 = transfac.read(fh)

        self.assertEqual(len(motifs), 1)
        npt.assert_array_almost_equal(motifs[0].probability_matrix, PWM, 2)
        npt.assert_array_almost_equal(motifs2[0].probability_matrix, PWM, 2)

        self.assertListEqual(motifs[0].alphabet.tolist(), list("ACGT"))
        self.assertEqual(motifs[0].alphabet_len, 4)
        self.assertAlmostEqual(motifs[0].information_content, 7.6535, 3)


class TestMotifProps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        meme_file_name = os.path.join(artifact_path, ECORI_FILE_NAME)
        cls.master_motif = meme.read(meme_file_name)[0]

    def setUp(self):

        self.motif = copy.deepcopy(self.master_motif)

    def test_information_content(self):

        self.assertEqual(self.motif.information_content, 24.0)

    def test_consensus(self):

        self.assertEqual(self.motif.consensus, "GAATTCGAATTC")

    def test_length(self):

        self.assertEqual(len(self.motif), 12)

    def test_score(self):

        self.assertEqual(self.motif.score_match("GAATTCGAATTC"), 24.0)
        self.assertEqual(self.motif.score_match("GAATTCGAATAC"), 18.0)
        self.assertEqual(self.motif.score_match("GAATTCGAATAA"), 12.0)
        self.assertEqual(self.motif.score_match("GAATTCGATACG"), 0.0)
        self.assertEqual(self.motif.score_match("GAATTCCTTAAG"), 0.0)
        self.assertEqual(self.motif.score_match("CTTAAGCTTAAG"), 0.0)

    def test_shuffle(self):

        npt.assert_array_almost_equal(self.motif.probability_matrix.sum(axis=1), np.ones(len(self.motif)))
        npt.assert_array_almost_equal(self.motif.probability_matrix, self.master_motif.probability_matrix)

        self.motif.shuffle(random_seed=100)

        npt.assert_array_almost_equal(self.motif.probability_matrix.sum(axis=1), np.ones(len(self.motif)))

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(self.motif.probability_matrix, self.master_motif.probability_matrix)

        motif_copy = copy.deepcopy(self.master_motif)
        motif_copy.shuffle(random_seed=100)

        npt.assert_array_almost_equal(self.motif.probability_matrix, motif_copy.probability_matrix)
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(motif_copy.probability_matrix, self.master_motif.probability_matrix)

        motif_copy = copy.deepcopy(self.master_motif)
        shuffle_motifs([motif_copy], 100)

        npt.assert_array_almost_equal(self.motif.probability_matrix, motif_copy.probability_matrix)
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(motif_copy.probability_matrix, self.master_motif.probability_matrix)
