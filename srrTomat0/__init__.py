import os

__version__ = '0.1.0'

# Paths to the executable files
# Defaults assume that executables are in the shell path

STAR_EXECUTABLE_PATH = os.path.expanduser("STAR")
PREFETCH_EXECUTABLE_PATH = os.path.expanduser("prefetch")
FASTQDUMP_EXECUTABLE_PATH = os.path.expanduser("fastq-dump")
CHROMA_EXECUTABLE_PATH = os.path.expanduser("ChromA")

HTSEQ_MODULE_NAME = "HTSeq.scripts.count"
