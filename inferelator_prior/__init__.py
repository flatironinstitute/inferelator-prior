import os

__version__ = '0.1.0'

# Paths to the executable files
# Defaults assume that executables are in the shell path

STAR_EXECUTABLE_PATH = os.path.expanduser("STAR")
PREFETCH_EXECUTABLE_PATH = os.path.expanduser("prefetch")
FASTQDUMP_EXECUTABLE_PATH = os.path.expanduser("fastq-dump")
CHROMA_EXECUTABLE_PATH = os.path.expanduser("ChromA")
SAMTOOLS_EXECUTABLE_PATH = os.path.expanduser("samtools")
FIMO_EXECUTABLE_PATH = os.path.expanduser("fimo")
HOMER_EXECUTABLE_PATH = os.path.expanduser("homer2")
BEDTOOLS_EXECUTABLE_PATH = os.path.expanduser("bedtools")

HTSEQ_MODULE_NAME = "HTSeq.scripts.count"
CHROMA_MODULE_NAME = "ChromA"

SRR_SUBPATH = "SRR"
FASTQ_SUBPATH = "FASTQ"
STAR_ALIGNMENT_SUBPATH = "STAR"
HTSEQ_ALIGNMENT_SUBPATH = "HTSEQ"
BAM_SUBPATH = "BAM"
FIMO_SUBPATH = "FIMO"
