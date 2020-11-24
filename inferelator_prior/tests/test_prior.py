import os
import unittest
import pandas as pd

from inferelator_prior.motifs import meme, motifs_to_dataframe, MotifScan, fimo, MOTIF_NAME_COL, SCAN_SCORE_COL
from inferelator_prior.processor import prior, gtf

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")

ECORI_FILE_NAME = os.path.join(artifact_path, "test_ecori.meme")
FASTA_FILE_NAME = os.path.join(artifact_path, "test_motif_search.fasta")
BED_FILE_NAME = os.path.join(artifact_path, "test_motif_search.bed")
GTF_FILE_NAME = os.path.join(artifact_path, "test_motif_search.gtf")
MOTIF_STARTS = [3, 35, 87, 199, 711]


class TestPriorPipeline(unittest.TestCase):

    def setUp(self):
        self.motifs = meme.read(ECORI_FILE_NAME)
        self.genes = gtf.load_gtf_to_dataframe(GTF_FILE_NAME)
        self.motif_information = motifs_to_dataframe(self.motifs)

    def test_prior_agg_by_base(self):
        motif_peaks, _, _ = self.do_scan_prior(20)
        motif_peaks[fimo.FIMO_START] = [7, 13, 19, 1]
        motif_peaks[fimo.FIMO_STOP] = [18, 24, 30, 12]
        motif_peaks[MOTIF_NAME_COL] = 'ECORI'

        agg_peaks = prior.MotifScorer._agg_per_base(motif_peaks)
        self.assertEqual(agg_peaks[fimo.FIMO_START].values[0], 1)
        self.assertEqual(agg_peaks[fimo.FIMO_STOP].values[0], 30)
        self.assertEqual(agg_peaks[prior.SCAN_SCORE_COL].values[0], 48)

        motif_peaks[fimo.FIMO_START] = [7, 13, 1, 19]
        motif_peaks[fimo.FIMO_STOP] = [18, 24, 12, 30]

        agg_peaks = prior.MotifScorer._agg_per_base(motif_peaks)

        self.assertEqual(agg_peaks[fimo.FIMO_START].values[0], 1)
        self.assertEqual(agg_peaks[fimo.FIMO_STOP].values[0], 30)
        self.assertEqual(agg_peaks[prior.SCAN_SCORE_COL].values[0], 60)

    def test_prior_no_tandem_20_window(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=0)
        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(20)
        self.assertEqual(motif_peaks.shape[0], 4)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 24.)

    def test_prior_no_tandem_200_window(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=0)
        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(200)
        self.assertEqual(motif_peaks.shape[0], 12)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 24.)

    def test_prior_50_tandem_200_window(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=50)
        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(200)

        self.assertEqual(motif_peaks.shape[0], 12)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 72.)

    def test_prior_50_tandem_10000_window(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=50)
        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(10000)
        self.assertEqual(motif_peaks.shape[0], 14)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 72.)

    def test_prior_no_tandem_1000_window_no_bed(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=0)
        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(1000, use_bed=False)

        self.assertEqual(motif_peaks.shape[0], 14)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 24.)

    def test_multiple_genes_50_tandem_100_window(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=50)
        self.genes = pd.concat((self.genes, pd.DataFrame({"seqname": "seq1",
                                                          "start": 550.,
                                                          "end": 750.,
                                                          "TSS": 750.,
                                                          "gene_name": "TEST2",
                                                          "strand": "-"}, index=[1])))

        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(100)
        self.assertEqual(motif_peaks.shape[0], 10)
        self.assertEqual(prior_matrix.sum().sum(), 2)

        tf = raw_matrix.iloc[:, 0]
        self.assertListEqual(tf.values.tolist(), [72., 24.])

    def test_multiple_genes_bad_chr(self):
        prior.MotifScorer.set_information_criteria(min_binding_ic=8, max_dist=50)
        self.genes = pd.concat((self.genes, pd.DataFrame({"seqname": "seq200",
                                                          "start": 550.,
                                                          "end": 750.,
                                                          "TSS": 750.,
                                                          "gene_name": "TEST2",
                                                          "strand": "-"}, index=[1])))

        motif_peaks, prior_matrix, raw_matrix = self.do_scan_prior(200)
        self.assertEqual(motif_peaks.shape[0], 12)
        self.assertEqual(prior_matrix.sum().sum(), 1)
        self.assertEqual(raw_matrix.max().max(), 72.)

    def test_matrix_cuts(self):
        info_matrix = pd.read_csv(os.path.join(artifact_path, "test_info_matrix.tsv.gz"), sep="\t", index_col=0)
        cut_matrix = prior.build_prior_from_motifs(info_matrix, num_workers=1)

        for i in info_matrix.columns:
            _is_called = cut_matrix[i] != 0
            _num_called = _is_called.sum()

            _kept = info_matrix[i][_is_called]
            _not_kept = info_matrix[i][~_is_called]

            self.assertGreater(_kept.min(), _not_kept.max()) if _num_called > 0 else True

    def do_scan_prior(self, window_size, do_threshold=False, use_bed=True, use_tss=True):
        fasta_record_lengths = gtf.get_fasta_lengths(FASTA_FILE_NAME)
        genes = gtf.open_window(self.genes, window_size=window_size, use_tss=use_tss,
                                fasta_record_lengths=fasta_record_lengths)
        self.gene_locs = genes.loc[:, [gtf.GTF_CHROMOSOME, gtf.SEQ_START, gtf.SEQ_STOP, gtf.GTF_STRAND]].copy()
        self.gene_locs[[gtf.SEQ_START, gtf.SEQ_STOP]] = self.gene_locs[[gtf.SEQ_START, gtf.SEQ_STOP]].astype(int)

        ms = MotifScan.scanner(motifs=self.motifs, num_workers=1)
        motif_peaks = ms.scan(FASTA_FILE_NAME,
                              constraint_bed_file=BED_FILE_NAME if use_bed else None,
                              promoter_bed=self.gene_locs,
                              min_ic=0, threshold=5e-4)

        raw_matrix, _ = prior.summarize_target_per_regulator(genes, motif_peaks, self.motif_information, num_workers=1)
        prior_matrix = prior.build_prior_from_motifs(raw_matrix, num_workers=1, do_threshold=do_threshold)

        return motif_peaks, prior_matrix, raw_matrix
