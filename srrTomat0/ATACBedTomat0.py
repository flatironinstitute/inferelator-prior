from srrTomat0.processor.gtf import load_gtf_to_dataframe, open_window, GTF_CHROMOSOME, SEQ_START, SEQ_STOP
from srrTomat0.processor.bedtools import load_bed_to_dataframe, merge_overlapping_peaks
from srrTomat0.processor.prior import build_prior_from_atac_motifs
from srrTomat0.motifs.fimo import fimo_scan
from srrTomat0.motifs import motifs_to_dataframe, meme

import argparse


def main():
    ap = argparse.ArgumentParser(description="Create a prior from open chromatin peaks and motif peaks")
    ap.add_argument("-m", "--motif", dest="motif", help="Motif MEME file", metavar="PATH", required=True)
    ap.add_argument("-a", "--atac", dest="atac", help="ATAC BED file", metavar="FILE", required=True)
    ap.add_argument("-f", "--fasta", dest="fasta", help="Genomic FASTA file", metavar="FILE", required=True)
    ap.add_argument("-g", "--gtf", dest="annotation", help="GTF Annotation File", metavar="PATH", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    ap.add_argument("-w", "--window", dest="window_size", help="Window around genes", type=int, default=0)
    ap.add_argument("-c", "--cpu", dest="cores", help="Number of cores", metavar="CORES", type=int, default=1)
    ap.add_argument("--tss", dest="tss", help="Use TSS for window", action='store_const', const=True, default=False)
    ap.add_argument("--minimum_information", dest="min_ic", help="Minimum information content",
                    metavar="BITS", type=int, default=8)

    args = ap.parse_args()

    _, prior_matrix = build_atac_motif_prior(args.motif, args.atac, args.annotation, args.fasta,
                                             window_size=args.window_size, num_cores=args.cores, use_tss=args.tss,
                                             min_ic=args.min_ic)
    
    prior_matrix.astype(int).to_csv(args.out, sep="\t")


def build_atac_motif_prior(motif_meme_file, atac_bed_file, annotation_file, genomic_fasta_file, window_size=0,
                           use_tss=True, motif_type='fimo', num_cores=1, min_ic=8):

    print("Loading genes from file ({f})".format(f=annotation_file))
    # Load genes and open a window
    genes = load_gtf_to_dataframe(annotation_file)
    genes = open_window(genes, window_size=window_size, use_tss=use_tss)
    print("\t{n} genes loaded".format(n=genes.shape[0]))

    print("Loading motifs from file ({f})".format(f=motif_meme_file))
    motif_information = motifs_to_dataframe(meme.read(motif_meme_file))

    # Load and scan target chromatin peaks
    print("Scanning target chromatin ({f_c}) for motifs ({f_m})".format(f_c=atac_bed_file, f_m=motif_meme_file))
    motif_peaks = fimo_scan(atac_bed_file, genomic_fasta_file, meme_file=motif_meme_file, num_workers=num_cores,
                            min_ic=min_ic)

    # Processing into prior
    print("Processing TF binding sites into prior")
    return build_prior_from_atac_motifs(genes, motif_peaks, motif_information)


if __name__ == '__main__':
    main()
