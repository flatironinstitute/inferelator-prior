import argparse

from inferelator_prior.network_from_motifs import load_and_process_motifs, fuzzy_merge_motifs
from inferelator_prior.motifs import (MOTIF_COL, MOTIF_NAME_COL, INFO_COL, ENTROPY_COL, LEN_COL, MOTIF_CONSENSUS_COL,
                                      MOTIF_ORIGINAL_NAME_COL)

OUTPUT_COLS = [MOTIF_COL, MOTIF_NAME_COL, INFO_COL, ENTROPY_COL, LEN_COL, MOTIF_CONSENSUS_COL]


def main():
    ap = argparse.ArgumentParser(description="Read a motif file and produce a summary TSV")

    # REQUIRED ARGUMENTS ###############################################################################################

    ap.add_argument("-m", "--motif", dest="motif", help="Motif file", metavar="FILE", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output TSV", metavar="FILE", default="./motif_information.tsv")
    ap.add_argument("--motif_format", dest="motif_format", help="Motif file FORMAT (transfac or meme)",
                    metavar="FORMAT", default="meme")
    ap.add_argument("--fuzzy", dest="fuzzy", help="Use fuzzy motif name merging", action='store_const',
                    const=True, default=False)

    args = ap.parse_args()

    summarize_motifs(args.motif, args.out, motif_format=args.motif_format, fuzzy=args.fuzzy)


def summarize_motifs(motif_file, output_file, motif_format="meme", fuzzy=False):

    _, motif_information = load_and_process_motifs(motif_file, motif_format, fuzzy=fuzzy)
    out_cols = OUTPUT_COLS + [MOTIF_ORIGINAL_NAME_COL] if fuzzy else OUTPUT_COLS

    if output_file is not None:
        motif_information[out_cols].to_csv(output_file, sep="\t", float_format='%.3f', index=False)

    return motif_information


if __name__ == '__main__':
    main()
