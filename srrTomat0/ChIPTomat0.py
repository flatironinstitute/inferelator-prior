import argparse
import os

import pandas as pd
import pybedtools
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Load peaks and genes.")
    ap.add_argument("-f", "--file", dest="file", help="bed file containing ChIP peaks", metavar="FILE", default=None)
    ap.add_argument("-a", "--annotation", dest="anno", help="GTF/GFF Annotation File", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)

    args = ap.parse_known_args()



def chip_tomat0(chip_peaks, output_path, annotation_file):
    output_path = file_path_abs(output_path)
    chip_peaks = pybedtools.BedTool('/Users/cskokgibbs/Dropbox (Simons Foundation)/Skok_Lab_mESC/CHIP/ctcf.peaks_C.ID.bed')
    chip_peaks = chip_peaks.to_dataframe()

    annotation_file = pybedtools.BedTool('/Users/cskokgibbs/Dropbox (Simons Foundation)/Skok_Lab_mESC/CHIP/mm10.gtf')
    annotation_file = annotation_file.to_dataframe()
    attributes = annotation_file['attributes'].str.extract('gene_id\s\"([A-Za-z0-9\.\-\(\)]+)\"\;', expand=False)
    annotation_file['gene_name'] = attributes
    genes = annotation_file[['seqname', 'start', 'end', 'gene_name']]
