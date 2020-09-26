import tempfile
import shutil
import os
import unittest

from inferelator_prior.processor.srr import get_srr_files, unpack_srr_files

TEST_SRR_IDS = ["SRR053325"]


class TestSRR(unittest.TestCase):

    srr_ids = TEST_SRR_IDS
    temp_path = None

    @classmethod
    def setUpClass(cls):
        cls.temp_path = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_path, ignore_errors=True)

    def test_srr_get_success(self):
        srr_files = get_srr_files(self.srr_ids, self.temp_path, prefetch_options=["--transport", "http"])
        self.assertTrue(os.path.exists(srr_files[0]))
        self.assertEqual(os.path.getsize(srr_files[0]), 31838)

        srr_files_2 = get_srr_files(self.srr_ids, self.temp_path)
        self.assertEqual(srr_files[0], srr_files_2[0])

    def test_srr_get_fail(self):
        self.assertIsNone(get_srr_files([""], os.path.join(self.temp_path, "blah", "blah"),
                                        prefetch_options=["--transport", "http"])[0])

    def test_srr_unpack_success(self):
        srr_files = get_srr_files(self.srr_ids, self.temp_path, prefetch_options=["--transport", "http"])
        fastq_files = unpack_srr_files(self.srr_ids, srr_files, self.temp_path)
        self.assertTrue(all(map(lambda x: os.path.exists(x), fastq_files[0])))
        self.assertEqual(len(fastq_files[0]), 3)

        fastq_files2 = unpack_srr_files(self.srr_ids, srr_files, self.temp_path)
        self.assertListEqual(fastq_files[0], fastq_files2[0])

    def test_srr_unpack_fail(self):
        self.assertListEqual(unpack_srr_files(self.srr_ids, [""], self.temp_path)[0], [None])

    def test_srr_unpack_skip(self):
        self.assertListEqual(unpack_srr_files(self.srr_ids, [None], self.temp_path)[0], [None])

