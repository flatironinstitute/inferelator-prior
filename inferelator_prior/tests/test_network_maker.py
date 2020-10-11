import os
import unittest
import pandas as pd

from inferelator_prior.motifs import meme, select_motifs, truncate_motifs
from inferelator_prior.processor import gtf

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.genes = gtf.load_gtf_to_dataframe(os.path.join(artifact_path, "sc64_mito.gtf"))
        self.motifs = meme.read(os.path.join(artifact_path, "test.meme"))
        self.motifs.extend(meme.read(os.path.join(artifact_path, "test_ecori.meme")))

    def test_gene_constrain_all(self):
        constraint_to = self.genes[gtf.GTF_GENENAME].unique().tolist()
        trim_genes = gtf.select_genes(self.genes, constraint_to)
        self.assertEqual(trim_genes.shape[0], self.genes.shape[0])
        self.assertCountEqual(constraint_to, trim_genes[gtf.GTF_GENENAME].tolist())

    def test_gene_constrain_all_good(self):
        constraint_to = ["Q0020", "Q0017", "Q0045"]
        trim_genes = gtf.select_genes(self.genes, constraint_to)
        self.assertEqual(trim_genes.shape[0], 3)
        self.assertCountEqual(constraint_to, trim_genes[gtf.GTF_GENENAME].tolist())

    def test_gene_constrain_some_good(self):
        constraint_to = ["Q0020", "Q0017", "Q0045", "YAL009C"]
        trim_genes = gtf.select_genes(self.genes, constraint_to)
        self.assertEqual(trim_genes.shape[0], 3)
        self.assertCountEqual(constraint_to[:3], trim_genes[gtf.GTF_GENENAME].tolist())

    def test_gene_constrain_no_good(self):
        constraint_to = ["YBL021W", "YGR121C", "YDL090W", "YAL009C"]

        with self.assertRaises(ValueError):
            trim_genes = gtf.select_genes(self.genes, constraint_to)

    def test_tf_constrain_all(self):
        constraint_to = ["Gata4", "ECORI"]

        trim_motifs = select_motifs(self.motifs, constraint_to)
        self.assertEqual(len(trim_motifs), 2)
        self.assertCountEqual(list(map(lambda x: x.motif_name, self.motifs)),
                              list(map(lambda x: x.motif_name, trim_motifs)))

    def test_tf_constrain_all_good(self):
        constraint_to = ["Gata4"]

        trim_motifs = select_motifs(self.motifs, constraint_to)
        self.assertEqual(len(trim_motifs), 1)
        self.assertCountEqual(constraint_to,
                              list(map(lambda x: x.motif_name, trim_motifs)))

    def test_tf_constrain_no_good(self):
        constraint_to = ["ATF4", "CREB"]

        with self.assertRaises(ValueError):
            trim_motifs = select_motifs(self.motifs, constraint_to)
