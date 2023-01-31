import pandas as pd
import os
import tqdm

from inferelator_prior.processor.bedtools import (
    link_bed_to_genes,
    BED_COLS,
    GENE_COL,
    BED_SIGNAL_COL
)

ENCODE_TF_COL = 'TF'
ENCODE_FILE_NAME_COL = 'FileName'
ENCODE_FILE_EXISTS_COL = 'FileExists'
ENCODE_ACCESSION_COL = 'Accession'

_ENCODE_ACCESSION_COL = 'File accession'

ENCODE_WARN_COL = 'WARNING'
ENCODE_NONCOMPLIANT_COL = 'NONCOMPLIANT'
ENCODE_ERROR_COL = 'ERROR'

ENCODE_BED_EXT = ".bed.gz"

EXTRA_ENCODE_COLS = [
    ENCODE_TF_COL,
    GENE_COL,
    BED_SIGNAL_COL,
    ENCODE_ACCESSION_COL
]


def load_encode_metadata(
    encode_metadata_file,
    encode_data_path,
    encode_experiment_suffix="-human"
):

    metadata = pd.read_csv(
        os.path.join(
            encode_data_path,
            encode_metadata_file
        ),
        sep="\t"
    )

    # Extract TF names from experiment target column
    metadata[ENCODE_TF_COL] = metadata.loc[:, 'Experiment target'].str.strip(
       encode_experiment_suffix
    ).tolist()

    # Add a filename column and a check to see if it exists column
    metadata[ENCODE_FILE_NAME_COL] = list(map(
        lambda x: os.path.join(encode_data_path, x + ENCODE_BED_EXT),
        metadata[_ENCODE_ACCESSION_COL]
    ))

    metadata[ENCODE_FILE_EXISTS_COL] = list(map(
        lambda x: os.path.exists(x),
        metadata[ENCODE_FILE_NAME_COL]
    ))

    # Convert verbose error and warning columns to bools
    metadata[ENCODE_ERROR_COL] = ~metadata['Audit ERROR'].isna()
    metadata[ENCODE_WARN_COL] = ~metadata['Audit WARNING'].isna()
    metadata[ENCODE_NONCOMPLIANT_COL] = ~metadata['Audit NOT_COMPLIANT'].isna()

    return metadata


def process_encode_bed_files(
    encode_metadata,
    gene_annotations,
    output_file=None,
    skip_error=True,
    skip_noncompliant=False,
    debug=False
):

    not_exist = 0
    qc_skip = 0

    n_bed_records = 0
    n_linked_records = 0

    linked_data = []

    for i, (_, record) in tqdm.tqdm(
        enumerate(encode_metadata.iterrows()),
        total=encode_metadata.shape[0]
    ):

        # Check failure modes
        if not record[ENCODE_FILE_EXISTS_COL]:
            not_exist += 1
            continue

        if skip_error and record[ENCODE_ERROR_COL]:
            qc_skip += 1
            continue

        if skip_noncompliant and record[ENCODE_NONCOMPLIANT_COL]:
            qc_skip += 1
            continue

        if debug:
            print(
                f"Loading {record[_ENCODE_ACCESSION_COL]} from "
                f"{record[ENCODE_FILE_NAME_COL]}"
            )

        n_before, n_after, linked_bed = link_bed_to_genes(
            record[ENCODE_FILE_NAME_COL],
            gene_annotations,
            None,
            window_size=None,
            narrowpeak_bed=True,
            dprint=lambda *x: x,
            non_gene_key=None,
            check_chromosomes=i == 0 or debug
        )

        n_bed_records += n_before
        n_linked_records += n_after

        linked_bed.insert(
            5,
            ENCODE_TF_COL,
            record[ENCODE_TF_COL]
        )

        linked_bed[ENCODE_ACCESSION_COL] = record[_ENCODE_ACCESSION_COL]

        linked_data.append(linked_bed)

    print(
        f"Processed {encode_metadata.shape[0]} ENCODE records: "
        f"{qc_skip} records removed for ENCODE QC status, "
        f"{not_exist} records removed because BED file did not exist, "
        f"{n_bed_records} BED file lines linked to "
        f"{gene_annotations.shape[0]} genes ({n_linked_records} successful)"
    )

    linked_data = pd.concat(
        linked_data,
        axis=0
    )

    if output_file is not None:
        linked_data[BED_COLS + EXTRA_ENCODE_COLS].to_csv(
            output_file,
            sep="\t",
            index=False
        )

    return linked_data
