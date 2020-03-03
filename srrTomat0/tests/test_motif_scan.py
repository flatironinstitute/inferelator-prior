import unittest
import os
import io
import pandas as pd

from srrTomat0.motifs import Motif, MotifScanner, fimo, homer

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")

FASTA_FILE_NAME = os.path.join(artifact_path, "test_motif_search.fasta")
BED_FILE_NAME = os.path.join(artifact_path, "test_motif_search.bed")


TEST_MOTIF_MATRIX = """\
0.0\t0.0\t1.0\t0.0
1.0\t0.0\t0.0\t0.0
1.0\t0.0\t0.0\t0.0
0.0\t0.0\t0.0\t1.0
0.0\t0.0\t0.0\t1.0
0.0\t1.0\t0.0\t0.0
0.0\t0.0\t1.0\t0.0
1.0\t0.0\t0.0\t0.0
1.0\t0.0\t0.0\t0.0
0.0\t0.0\t0.0\t1.0
0.0\t0.0\t0.0\t1.0
0.0\t1.0\t0.0\t0.0
"""

MOTIF_OBJ = Motif("EcoRI", "EcoRI", list("ACGT"))
MOTIF_OBJ.probability_matrix = pd.read_csv(io.StringIO(TEST_MOTIF_MATRIX), sep="\t", header=None, index_col=None)\
    .astype(float)\
    .values

MOTIF_STARTS = [3, 35, 87, 199, 711]


class TestScan(unittest.TestCase):

    def test_base(self):

        scanner = MotifScanner(motifs=[MOTIF_OBJ], num_workers=1)

        with self.assertRaises(NotImplementedError):
            scanner.scan(BED_FILE_NAME, FASTA_FILE_NAME, min_ic=8)

        with self.assertRaises(NotImplementedError):
            scanner._preprocess(8)

        with self.assertRaises(NotImplementedError):
            scanner._parse_output(None)

        with self.assertRaises(NotImplementedError):
            scanner._get_motifs(None, None)

    def test_fimo(self):
        scanner = fimo.FIMOScanner(motifs=[MOTIF_OBJ], num_workers=1)
        motif_locs = scanner.scan(BED_FILE_NAME, FASTA_FILE_NAME, min_ic=8)
        self.assertEqual(motif_locs.shape[0], 10)
        self.assertEqual(motif_locs.loc[motif_locs[fimo.FIMO_STRAND] == "+", :].shape[0], 5)
        self.assertEqual(motif_locs.loc[motif_locs[fimo.FIMO_STRAND] == "-", :].shape[0], 5)

        self.assertListEqual(motif_locs.loc[motif_locs[fimo.FIMO_STRAND] == "+", fimo.FIMO_START].tolist(),
                             MOTIF_STARTS)
        self.assertListEqual(motif_locs.loc[motif_locs[fimo.FIMO_STRAND] == "-", fimo.FIMO_START].tolist(),
                             MOTIF_STARTS)

    def test_homer(self):
        scanner = homer.HOMERScanner(motifs=[MOTIF_OBJ], num_workers=1)
        motif_locs = scanner.scan(BED_FILE_NAME, FASTA_FILE_NAME, min_ic=8)
        print(motif_locs[homer.HOMER_START])
        self.assertEqual(motif_locs.shape[0], 10)
        self.assertEqual(motif_locs.loc[motif_locs[homer.HOMER_STRAND] == "+", :].shape[0], 5)
        self.assertEqual(motif_locs.loc[motif_locs[homer.HOMER_STRAND] == "-", :].shape[0], 5)

        self.assertListEqual(motif_locs.loc[motif_locs[homer.HOMER_STRAND] == "+", homer.HOMER_START]
                             .tolist(),
                             MOTIF_STARTS)
        self.assertListEqual(motif_locs.loc[motif_locs[homer.HOMER_STRAND] == "-", homer.HOMER_START].sort_values()
                             .tolist(),
                             MOTIF_STARTS)
