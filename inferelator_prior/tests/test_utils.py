import os
import shutil
import tempfile
import unittest
from urllib.error import URLError

import inferelator_prior.processor.utils as utils


class TestUtils(unittest.TestCase):
    temp_path = None

    @classmethod
    def setUpClass(cls):
        cls.temp_path = tempfile.mkdtemp()
        cls.star_ref_path = os.path.join(cls.temp_path, "star")
        cls.sam_out_path = os.path.join(cls.temp_path, "sam")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_path, ignore_errors=True)

    def test_ncbi_genomes(self):
        fasta, gtf = utils.get_genome_file_locs("sc64")
        self.assertEqual(gtf[1], "sc64.gtf.gz")

        with self.assertRaises(ValueError):
            utils.get_genome_file_locs("not_a_real_thing")

    @unittest.skipIf("TRAVIS_PYTHON_VERSION" in os.environ, "Skipping URL test on TRAVIS")
    def test_get_file_from_url(self):
        fasta, gtf = utils.get_genome_file_locs("sc64")
        target_path = os.path.join(self.temp_path, "test.gtf.gz")
        file_path = utils.get_file_from_url(gtf[0], target_path)
        self.assertEqual(file_path, target_path)
        self.assertTrue(os.path.exists(file_path))

        with self.assertRaises(URLError):
            file_path = utils.get_file_from_url(gtf[0] + "does_not_exist.file", target_path)

    def test_requirements(self):
        self.assertTrue(utils.test_requirements_exist(test_targets=["python"],
                                                      test_package={"python": ("python", ["python", "--version"])},
                                                      test_htseq=False))

        self.assertTrue(utils.test_requirements_exist(test_targets=["python"],
                                                      test_package={"python": ("python", ["python", "--version"])},
                                                      test_htseq=True))

        with self.assertRaises(FileNotFoundError):
            utils.test_requirements_exist(test_targets=["not_a-.thing"],
                                          test_package={"not_a-.thing": ("fake", ["not_a-.thing", "--version"])},
                                          test_htseq=False)

    def test_file_path_abs(self):
        self.assertEqual(os.path.abspath(os.path.expanduser("~")), utils.file_path_abs("~"))
