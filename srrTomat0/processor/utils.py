import os
import subprocess
import shutil
import argparse
import sys

if sys.version_info[0] < 3:
    print("The srrTomat0 package requires python3")
    exit(1)

import urllib.parse
import urllib.request

from srrTomat0 import STAR_EXECUTABLE_PATH, PREFETCH_EXECUTABLE_PATH, FASTQDUMP_EXECUTABLE_PATH

# Tuple of ((fasta_url, fasta_file_name), (gff_url, gff_file_name))

_HG38 = (("ftp://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
          "hg38.fa.gz"),
         ("ftp://ftp.ensembl.org/pub/release-97/gff3/homo_sapiens/Homo_sapiens.GRCh38.97.gff3.gz",
          "hg38.gff3.gz"))

_SC64 = (("ftp://ftp.ensembl.org/pub/release-97/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz",
          "sc64.fa.gz"),
         ("ftp://ftp.ensembl.org/pub/release-97/gff3/saccharomyces_cerevisiae/Saccharomyces_cerevisiae.R64-1-1.97.gff3.gz",
          "sc64.gff3.gz"))

# Key by genome name
_DEFAULT_GENOMES = {"hg38": _HG38, "sc64": _SC64}

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


def test_requirements_exist(test_package=_TEST_REQUIREMENTS, test_htseq=True):

    ret_code = {}
    for req, (pref, cmd) in test_package.items():
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret_code[req] = proc.returncode
        stdout = pref + " ".join([l for l in proc.stdout.decode().strip().split("\n") if l.strip() != ""])
        print(stdout)

    failed = False
    for req, code in ret_code.items():
        if code != 0:
            print("{req} not found [{args}]".format(req=req, args=" ".join(test_package[req])))
            failed = True

    if test_htseq:
        try:
            import HTSeq
            print("HTSeq : " + str(HTSeq.__version__))
        except ImportError:
            print("HTSeq : HTSeq not found (ImportError)")
            failed=True

    if failed:
        raise FileNotFoundError


# ArgumentParser that tests requirements if it fails to parse arguments
class ArgParseTestRequirements(argparse.ArgumentParser):

    def error(self, message):
        try:
            test_requirements_exist()
        except FileNotFoundError:
            pass
        finally:
            super(ArgParseTestRequirements, self).error(message)
