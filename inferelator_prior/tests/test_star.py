import os
import tempfile
import unittest
import shutil

import pysam

from inferelator_prior.processor.star import star_mkref, star_align_fastqs

GENOME_FILE_NAME = "sc64_mito.fasta"
ANNOTATION_FILE_NAME = "sc64_mito.gtf"
TEST_FASTQ = "test_alignment.fastq.gz"

STAR_REF_FILES = ["chrLength.txt", "chrName.txt", "exonGeTrInfo.tab", "geneInfo.tab", "genomeParameters.txt", "SA",
                  "sjdbInfo.txt", "sjdbList.out.tab", "chrNameLength.txt", "chrStart.txt", "exonInfo.tab", "Genome",
                  "SAindex", "sjdbList.fromGTF.out.tab", "transcriptInfo.tab"]

artifact_path = os.path.join(os.path.abspath(os.path.expanduser(os.path.dirname(__file__))), "artifacts")


class TestSTAR(unittest.TestCase):
    genome_file = os.path.join(artifact_path, GENOME_FILE_NAME)
    annotation_file = os.path.join(artifact_path, ANNOTATION_FILE_NAME)
    fastq_file = os.path.join(artifact_path, TEST_FASTQ)

    temp_path = None
    star_ref_path = None
    sam_out_path = None

    @classmethod
    def setUpClass(cls):
        cls.temp_path = tempfile.mkdtemp()
        cls.star_ref_path = os.path.join(cls.temp_path, "star")
        cls.sam_out_path = os.path.join(cls.temp_path, "sam")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_path, ignore_errors=True)

    def test_star(self):
        self._step_mkref()
        sam_files = self._step_align()
        self._step_test_alignment(sam_files)

    def _step_mkref(self):
        out_path = star_mkref(self.star_ref_path, genome_file=[self.genome_file], annotation_file=self.annotation_file,
                              move_files=False)

        self.assertFalse(out_path is None)
        self.assertTrue(out_path == self.star_ref_path)

        for file_name in STAR_REF_FILES:
            self.assertTrue(os.path.exists(os.path.join(out_path, file_name)))

    def _step_align(self):
        sam_files = star_align_fastqs(["TEST"], [[self.fastq_file]], self.star_ref_path, self.sam_out_path)

        self.assertFalse(sam_files[0] is None)

        return sam_files

    def _step_test_alignment(self, sam_files):
        samfile = pysam.AlignmentFile(sam_files[0], "r")
        reads = [aln for aln in samfile.fetch()]

        self.assertEqual(len(reads), 9)
        self.assertEqual(sum(map(lambda x: x.is_reverse, reads)), 3)

        self.assertListEqual(list(map(lambda x: x.reference_start, reads)),
                             [28620, 30480, 3078, 4893, 5465, 5545, 37969, 37969, 38033])
        self.assertListEqual(list(map(lambda x: x.reference_end, reads)),
                             [28719, 30579, 3177, 4992, 5564, 5644, 38068, 38068, 38132])
