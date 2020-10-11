import os
import unittest
import pandas as pd

from inferelator_prior.motifs import meme, motifs_to_dataframe, MotifScan, fimo, MOTIF_NAME_COL, SCAN_SCORE_COL
from inferelator_prior.processor import prior, gtf

class TestConstraints(unittest.TestCase):

    def test_gene_constrain(self):
        pass