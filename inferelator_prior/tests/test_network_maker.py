import os
import unittest
import pandas.testing as pdt
import tempfile

from inferelator_prior.motifs import meme, select_motifs
from inferelator_prior.processor import gtf
from inferelator_prior.network_from_motifs import load_and_process_motifs, build_motif_prior_from_genes
from inferelator_prior.network_from_motifs_fasta import build_motif_prior_from_fasta
from inferelator_prior.motif_information import summarize_motifs

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")
data_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "../../data/")
temppath = tempfile.TemporaryDirectory(prefix="ip_test_")
temp_path_prefix = os.path.join(temppath.name, "prior")

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
                                                   window_size=(500, 100),
                                                   intergenic_only=False,
                                                   output_prefix=None)

        self.assertEqual(cut.sum().sum(), 3)
        self.assertListEqual(cut[cut["GAL4"]].index.tolist(), ["YBR018C", "YBR019C", "YBR020W"])
        self.assertEqual((raw > 0).sum().sum(), 3)

    def test_full_stack_network_build_highmem(self):
        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                                                "test_gal4.meme"),
                                                   os.path.join(artifact_path,
                                                                "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                                                   os.path.join(data_path,
                                                                "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                                                   window_size=(500, 100),
                                                   intergenic_only=False,
                                                   output_prefix=None,
                                                   lowmem=False)

        self.assertEqual(cut.sum().sum(), 3)
        self.assertListEqual(cut[cut["GAL4"]].index.tolist(), ["YBR018C", "YBR019C", "YBR020W"])
        self.assertEqual((raw > 0).sum().sum(), 3)

    def test_full_stack_network_build_fasta(self):
        cut, raw, _ = build_motif_prior_from_fasta(os.path.join(artifact_path, "test_ecori.meme"),
                                                   os.path.join(artifact_path, "test_motif_search.fasta"),
                                                   output_prefix=None,
                                                   num_cores=1)

        self.assertEqual(cut.sum().sum(), 1)
        self.assertListEqual(cut[cut["EcoRI"]].index.tolist(), ["seq1"])
        self.assertEqual((raw > 0).sum().sum(), 1)

    def test_file_output(self):
        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                                        "test_gal4.meme"),
                                            os.path.join(artifact_path,
                                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                                            os.path.join(data_path,
                                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                                            window_size=(500, 100),
                                            intergenic_only=False,
                                            output_prefix=temp_path_prefix)

        self.assertTrue(os.path.exists(temp_path_prefix + "_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "_edge_matrix.tsv.gz"))
        self.assertFalse(os.path.exists(temp_path_prefix + "_tf_binding_locs.tsv"))
        self.assertFalse(os.path.exists(temp_path_prefix + "_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                                "test_gal4.meme"),
                                    os.path.join(artifact_path,
                                                "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                                    os.path.join(data_path,
                                                "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                                    window_size=(500, 100),
                                    intergenic_only=False,
                                    output_prefix=temp_path_prefix + "b",
                                    save_locs=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "b_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "b_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "b_tf_binding_locs.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                        "test_gal4.meme"),
                            os.path.join(artifact_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                            os.path.join(data_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                            window_size=(500, 100),
                            intergenic_only=False,
                            output_prefix=temp_path_prefix + "c",
                            save_locs=True,
                            lowmem=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "c_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "c_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "c_tf_binding_locs.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                        "test_gal4.meme"),
                            os.path.join(artifact_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                            os.path.join(data_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                            window_size=(500, 100),
                            intergenic_only=False,
                            output_prefix=temp_path_prefix + "d",
                            save_locs_filtered=True,
                            lowmem=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "d_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "d_edge_matrix.tsv.gz"))
        self.assertFalse(os.path.exists(temp_path_prefix + "d_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "d_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                        "test_gal4.meme"),
                            os.path.join(artifact_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                            os.path.join(data_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                            window_size=(500, 100),
                            intergenic_only=False,
                            output_prefix=temp_path_prefix + "e",
                            save_locs_filtered=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "e_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "e_edge_matrix.tsv.gz"))
        self.assertFalse(os.path.exists(temp_path_prefix + "e_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "e_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                "test_gal4.meme"),
                            os.path.join(artifact_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                            os.path.join(data_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                            window_size=(500, 100),
                            intergenic_only=False,
                            output_prefix=temp_path_prefix + "f",
                            save_locs_filtered=True,
                            save_locs=True,
                            lowmem=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "f_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_genes(os.path.join(artifact_path,
                                "test_gal4.meme"),
                            os.path.join(artifact_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.GAL_OPERON.gtf"),
                            os.path.join(data_path,
                                        "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"),
                            window_size=(500, 100),
                            intergenic_only=False,
                            output_prefix=temp_path_prefix + "g",
                            save_locs_filtered=True,
                            save_locs=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "g_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "g_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "g_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "g_tf_binding_locs_filtered.tsv"))


    def test_file_output_fasta(self):
        cut, raw, _ = build_motif_prior_from_fasta(os.path.join(artifact_path, "test_ecori.meme"),
                                                   os.path.join(artifact_path, "test_motif_search.fasta"),
                                                   output_prefix=temp_path_prefix)

        self.assertTrue(os.path.exists(temp_path_prefix + "_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "_edge_matrix.tsv.gz"))
        self.assertFalse(os.path.exists(temp_path_prefix + "_tf_binding_locs.tsv"))
        self.assertFalse(os.path.exists(temp_path_prefix + "_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_fasta(os.path.join(artifact_path, "test_ecori.meme"),
                                                   os.path.join(artifact_path, "test_motif_search.fasta"),
                            output_prefix=temp_path_prefix + "c",
                            save_locs=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "c_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "c_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "c_tf_binding_locs.tsv"))

        cut, raw, _ = build_motif_prior_from_fasta(os.path.join(artifact_path, "test_ecori.meme"),
                                                   os.path.join(artifact_path, "test_motif_search.fasta"),
                            output_prefix=temp_path_prefix + "d",
                            save_locs_filtered=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "d_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "d_edge_matrix.tsv.gz"))
        self.assertFalse(os.path.exists(temp_path_prefix + "d_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "d_tf_binding_locs_filtered.tsv"))

        cut, raw, _ = build_motif_prior_from_fasta(os.path.join(artifact_path, "test_ecori.meme"),
                                                   os.path.join(artifact_path, "test_motif_search.fasta"),
                            output_prefix=temp_path_prefix + "f",
                            save_locs_filtered=True,
                            save_locs=True)

        self.assertTrue(os.path.exists(temp_path_prefix + "f_unfiltered_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_edge_matrix.tsv.gz"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_tf_binding_locs.tsv"))
        self.assertTrue(os.path.exists(temp_path_prefix + "f_tf_binding_locs_filtered.tsv"))
