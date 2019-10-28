from srrTomat0.processor.gtf import load_gtf_to_dataframe, open_window, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.processor.bedtools import load_bed_to_dataframe, merge_overlapping_peaks
from srrTomat0.processor.motif_locations import MotifLocationManager as MotifLM
from srrTomat0.processor.prior import build_prior_from_atac_motifs

import argparse


def main():
    ap = argparse.ArgumentParser(description="Create a prior from open chromatin peaks and motif peaks")
    ap.add_argument("-m", "--motif", dest="motif", help="Motif BED file", metavar="PATH", required=True)
    ap.add_argument("-a", "--atac", dest="atac", help="ATAC BED file", metavar="FILE", required=True)
    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="PATH", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=0)
    ap.add_argument("-c", "--cpu", dest="cores", help="Number of cores", metavar="CORES", type=int, default=1)
    ap.add_argument("--tss", dest="tss", help="Use TSS for window", action='store_const', const=True, default=False)

    args = ap.parse_args()

    _, prior_matrix = build_atac_motif_prior(args.motif, args.atac, args.annotation, window_size=args.window_size,
                                             num_cores=args.cores, use_tss=args.tss)
    prior_matrix.to_csv(args.out, sep="\t")


def build_atac_motif_prior(motif_bed_file, atac_bed_file, annotation_file, window_size=0, use_tss=True,
                           motif_type='fimo', num_cores=1, fdr_alpha=0.05):

    MotifLM.set_motif_file_type(motif_type)
    motif_peaks = MotifLM.get_motifs(motif_bed_file)

    print("Loading genes from file ({f})".format(f=annotation_file))
    # Load genes and open a window
    genes = load_gtf_to_dataframe(annotation_file)
    genes = open_window(genes, window_size=window_size, use_tss=use_tss)
    print("\t{n} genes loaded".format(n=genes.shape[0]))

    # Load and merge open chromatin peaks
    print("Loading Open Chromatin Peaks from file ({f})".format(f=atac_bed_file))
    open_chromatin = load_bed_to_dataframe(atac_bed_file, header=None, names=[GTF_CHROMOSOME, SEQ_START, SEQ_STOP])
    print("\t{n} peaks loaded".format(n=open_chromatin.shape[0]))
    open_chromatin = merge_overlapping_peaks(open_chromatin, strand_column=None, max_distance=1)
    print("\t{n} peaks remain after merge".format(n=open_chromatin.shape[0]))

    return build_prior_from_atac_motifs(genes, open_chromatin, motif_peaks, num_cores=num_cores, fdr_alpha=fdr_alpha)


if __name__ == '__main__':
    main()
