import pandas as pd
import argparse

from inferelator.utils import is_string

from .processor.bedtools import (
    load_bed_to_bedtools,
    intersect_bed,
    BED_CHROMOSOME,
    SEQ_START,
    SEQ_STOP
)

from .processor.encode import (
    ENCODE_TF_COL,
    GENE_COL,
    BED_SIGNAL_COL
)

from .processor.prior import build_prior_from_motifs

ENCODE_PRIOR_COLS = [
    BED_CHROMOSOME,
    SEQ_START,
    SEQ_STOP,
    ENCODE_TF_COL,
    GENE_COL,
    BED_SIGNAL_COL
]


def main():

    ap = argparse.ArgumentParser(
        description="Generate a ENCODE-derived prior"
    )

    ap.add_argument(
        "-l",
        "--linked_bed_files",
        dest="linked",
        help="Linked TF/Gene BED File",
        metavar="FILE",
        required=True,
        nargs="+"
    )

    ap.add_argument(
        "-m",
        "--mask_bed_file",
        dest="mask",
        help="Masking BED File",
        metavar="FILE",
        required=True
    )

    ap.add_argument(
        "-o",
        "--out",
        dest="out",
        help="Output File",
        metavar="FILE",
        default="prior.tsv.gz"
    )

    args = ap.parse_args()

    long_df, wide_df = encode_prior(
        args.linked,
        args.mask,
        return_long=args.wide
    )

    wide_df.to_csv(
        args.out,
        sep="\t",
        index=False
    )


def encode_prior(
    linked_bed_files,
    mask_bed_file,
    overlap_requirement=0.75,
    do_thresholding=True,
    random_seed=100
):
    """
    Generate an ENCODE prior by taking ENCODE TF-ChIP peaks linked to genes
    and masking out anything that doesn't intersect a BED file with putative
    regulatory regions for a cell type

    :param linked_bed_files: BED file(s) with TF and Gene linkages called
    :type linked_bed_files: list(str)
    :param mask_bed_file: BED file with regulatory regions to consider
    :type mask_bed_file: str
    :param overlap_requirement: How much of the TF-ChIP peak has to overlap
        with the regulatory region, defaults to 0.75
    :type overlap_requirement: float, optional
    :return: Long TF
    :rtype: _type_
    """

    if (
        is_string(linked_bed_files) or
        isinstance(linked_bed_files, pd.DataFrame)
    ):
        linked_bed_files = [linked_bed_files]

    mask_bed = load_bed_to_bedtools(mask_bed_file)

    all_prior = []

    for linked_file in linked_bed_files:

        peak_bed = load_bed_to_bedtools(linked_file)

        intersected_bed = intersect_bed(
            peak_bed,
            mask_bed,
            wa=True,
            f=overlap_requirement
        ).to_dataframe(
            names=linked_file.columns
        )

        intersected_bed = intersected_bed.drop_duplicates(
            subset=ENCODE_PRIOR_COLS
        )

        all_prior.append(intersected_bed)

        del intersected_bed

    all_prior = pd.concat(
        all_prior,
        axis=0
    )

    wide_prior = all_prior.pivot_table(
        values=BED_SIGNAL_COL,
        index=GENE_COL,
        columns=ENCODE_TF_COL,
        fill_value=0,
        aggfunc='sum'
    )

    wide_prior.index.name = None
    wide_prior.columns.name = None

    wide_prior = build_prior_from_motifs(
        wide_prior,
        num_workers=None,
        seed=random_seed,
        do_threshold=do_thresholding
    ).astype(int)

    return all_prior, wide_prior


if __name__ == '__main__':
    main()
