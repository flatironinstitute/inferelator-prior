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
    GENE_COL
)

ENCODE_PRIOR_COLS = [
    BED_CHROMOSOME,
    SEQ_START,
    SEQ_STOP,
    ENCODE_TF_COL,
    GENE_COL
]


def main():

    ap = argparse.ArgumentParser(
        description="Link ATAC peaks in a BED file to genes in a GTF file"
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

    ap.add_argument(
        "-w",
        "--wide",
        dest="wide",
        help="Return wide [genes x TFs] matrix",
        action='store_const', const=False, default=True
    )

    args = ap.parse_args()

    long_df = encode_prior(
        args.linked,
        args.mask,
        return_long=args.wide
    )

    long_df.to_csv(
        args.out,
        sep="\t",
        index=False
    )


def encode_prior(
    linked_bed_files,
    mask_bed_file,
    overlap_requirement=0.75,
    return_all_cols=False,
    return_wide=True
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

    if is_string(linked_bed_files):
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

        if not return_all_cols:
            intersected_bed = intersected_bed[ENCODE_PRIOR_COLS]

        intersected_bed = intersected_bed.drop_duplicates(
            subset=ENCODE_PRIOR_COLS
        )

        all_prior.append(intersected_bed)

        del intersected_bed

    all_prior = pd.concat(
        all_prior,
        axis=0
    )

    if return_wide:

        all_prior = all_prior[[GENE_COL, ENCODE_TF_COL]].drop_duplicates()
        all_prior['Fill'] = 1

        all_prior = all_prior.pivot_table(
            values='Fill',
            index=GENE_COL,
            columns=ENCODE_TF_COL,
            fill_value=0
        ).astype(bool).astype(int)

        all_prior.index.name = None
        all_prior.columns.name = None

    return all_prior


if __name__ == '__main__':
    main()
