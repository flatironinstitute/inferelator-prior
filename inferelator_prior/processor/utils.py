import argparse
import os
import shutil
import subprocess
import sys
import numpy as np

if sys.version_info[0] < 3:
    print("The inferelator_prior package requires python3")
    exit(1)

import urllib.parse
import urllib.request

from inferelator_prior import STAR_EXECUTABLE_PATH, PREFETCH_EXECUTABLE_PATH, FASTQDUMP_EXECUTABLE_PATH

# Tuple of ((fasta_url, fasta_file_name), (gff_url, gff_file_name))

_HG38 = ((
         "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.fna.gz",
         "hg38.fa.gz"),
         (
         "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gtf.gz",
         "hg38.gtf.gz"))

_SC64 = (
("ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz",
 "sc64.fa.gz"),
("ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.gtf.gz",
 "sc64.gtf.gz"))

_MM10 = ((
         "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_genomic.fna.gz",
         "mm10.fa.gz"),
         (
         "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_genomic.gtf.gz",
         "mm10.gtf.gz"))

# Key by genome name
_DEFAULT_GENOMES = {"hg38": _HG38, "sc64": _SC64, "mm10": _MM10}

# Requirements tests (produce version for each requirement)
_TEST_REQUIREMENTS = {'prefetch': ("", [PREFETCH_EXECUTABLE_PATH, "--version"]),
                      'fastq-dump': ("", [FASTQDUMP_EXECUTABLE_PATH, "--version"]),
                      'STAR': ("STAR : ", [STAR_EXECUTABLE_PATH, "--version"])}


def get_genome_file_locs(genome):
    if genome in _DEFAULT_GENOMES.keys():
        return _DEFAULT_GENOMES[genome]
    else:
        raise ValueError("Genome must be one of {k}".format(k=" ".join(_DEFAULT_GENOMES.keys())))


def get_file_from_url(file_url, file_name_local=None):
    """
    Download a file from a url to a local file
    :param file_url:
    :param file_name_local:
    :return:
    """

    if file_name_local is None:
        file_name_local = file_path_abs(urllib.parse.urlsplit(file_url).path.split("/")[-1])

    print("Downloading {url} to {file}".format(url=file_url, file=file_name_local))

    with urllib.request.urlopen(file_url) as remote_handle, open(file_name_local, mode="wb") as local_handle:
        shutil.copyfileobj(remote_handle, local_handle)

    return file_name_local


def file_path_abs(file_path):
    """
    Convert a file path to a safe absolute path
    :param file_path: str
    :return: str
    """
    return os.path.abspath(os.path.expanduser(file_path))


def test_requirements_exist(test_targets=_TEST_REQUIREMENTS.keys(), test_package=_TEST_REQUIREMENTS, test_htseq=True):
    """
    Test that the requirements to run this package exist. Print versions of what can be found and raise a ValueError if
    any required software is missing.
    :param test_targets: list(str)
        A list of packages to test
    :param test_package: dict
        A dict, keyed by package names, of the commands to run to test the package versions
    :param test_htseq: bool
        Test for the python HTSeq package
    :return:
    """

    failed = False

    for req, (pref, cmd) in {k: test_package[k] for k in test_targets}.items():
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = pref + " ".join([l for l in proc.stdout.decode().strip().split("\n") if l.strip() != ""])
            print(stdout)
        except subprocess.CalledProcessError:
            print("{req} : {req} not found (CalledProcessError)".format(req=req))
            failed = True

    if test_htseq:
        try:
            import HTSeq
            print("HTSeq : " + str(HTSeq.__version__))
        except ImportError:
            print("HTSeq : HTSeq not found (ImportError)")
            failed = True

    if failed:
        raise FileNotFoundError

    return True


# ArgumentParser that tests requirements if it fails to parse arguments
# I just want to run a script with no arguments and have it test dependencies
class ArgParseTestRequirements(argparse.ArgumentParser):

    def error(self, message):
        try:
            test_requirements_exist()
        except FileNotFoundError:
            pass
        finally:
            super(ArgParseTestRequirements, self).error(message)
