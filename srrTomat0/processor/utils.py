import os
import subprocess
import shutil
import urllib.parse
import urllib.request

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


def test_requirements_exist():

    code = subprocess.call(["prefetch", "--version"])
    if code != 0:
        raise FileNotFoundError("prefetch executable not found [prefetch <args>]")

    code = subprocess.call(["fastq-dump", "--version"])
    if code != 0:
        raise FileNotFoundError("fastq-dump executable not found [fastq-dump <args>]")

    code = subprocess.call(["STAR", "--version"])
    if code != 0:
        raise FileNotFoundError("STAR executable not found [STAR <args>]")

    code = subprocess.call(["python", "-c", "'import HTSeq.scripts.count'"])
    if code != 0:
        raise FileNotFoundError("HTSeq.scripts.count is not available to python [python -m HTSeq.scripts.count <args>]")

