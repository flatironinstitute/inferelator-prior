import os

# Paths to the executable files
# Defaults assume that executables are in the shell path

STAR_EXECUTABLE_PATH = os.path.expanduser("STAR")
PREFETCH_EXECUTABLE_PATH = os.path.expanduser("prefetch")
FASTQDUMP_EXECUTABLE_PATH = os.path.expanduser("fastq-dump")
SAMTOOLS_EXECUTABLE_PATH = os.path.expanduser("samtools")
FIMO_EXECUTABLE_PATH = os.path.expanduser("fimo")
HOMER_EXECUTABLE_PATH = os.path.expanduser("homer2")
BEDTOOLS_EXECUTABLE_PATH = os.path.expanduser("bedtools")
KALLISTO_EXECUTABLE_PATH = os.path.expanduser("kallisto")


HTSEQ_MODULE_NAME = "HTSeq.scripts.count"

SRR_SUBPATH = "SRR"
FASTQ_SUBPATH = "FASTQ"
STAR_ALIGNMENT_SUBPATH = "STAR"
KALLISTO_ALIGNMENT_SUBPATH = "KALLISTO"
HTSEQ_ALIGNMENT_SUBPATH = "HTSEQ"
BAM_SUBPATH = "BAM"
FIMO_SUBPATH = "FIMO"
