import os
import unittest
import pandas.testing as pdt

from inferelator_prior.motifs import meme, select_motifs, truncate_motifs
from inferelator_prior.processor import gtf
from inferelator_prior.network_from_motifs import (network_scan, network_build, load_and_process_motifs,
                                                   build_motif_prior_from_genes)
from inferelator_prior.motif_information import summarize_motifs

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")
data_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "../../data/")


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


class TestFullStack(unittest.TestCase):

    def test_load_motifs(self):
        m, minfo = load_and_process_motifs(os.path.join(artifact_path, "test_gal4.meme"), "meme",
                                           regulator_constraint_list=["GAL4"])

        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].motif_name, "GAL4")

        with self.assertRaises(ValueError):
            m, minfo = load_and_process_motifs(os.path.join(artifact_path, "test_gal4.meme"), "meme",
                                               regulator_constraint_list=["NOT_GAL4"])

        minfo_2 = summarize_motifs(os.path.join(artifact_path, "test_gal4.meme"), None)

        pdt.assert_frame_equal(minfo, minfo_2)

    def test_full_stack_network_build(self):
        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                                                "test_gal4.meme"),
                                                   os.path.join(artifact_path,
                                                                "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                                                   os.path.join(data_path,
                                                                "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                                                   window_size=(500, 100))

        self.assertEqual(cut.sum().sum(), 3)
        self.assertListEqual(cut[cut["GAL4"]].index.tolist(), ["YBR018C", "YBR019C", "YBR020W"])
        self.assertEqual((raw > 0).sum().sum(), 3)
