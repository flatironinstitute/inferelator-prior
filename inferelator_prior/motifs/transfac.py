from inferelator_prior.motifs import Motif

import numpy as np

TRANSFAC_CODES = {"AC": "Accession",
                  "ID": "ID",
                  "NA": "Name",
                  "DT": "Date",
                  "CO": "Copyright",
                  "DE": "Description",
                  "TY": "Type",
                  "OS": "",
                  "OL": "",
                  "BF": "Species",
                  "P0": "Alphabet",
                  "SR": "",
                  "BA": "",
                  "CC": "",
                  "PR": "Profile"}


def read(file_descript):

    # Parse if it's a string
    if isinstance(file_descript, str):
        with open(file_descript) as motif_fh:
            return [m for m in _parse_transfac_file(motif_fh)]

    # Parse if it's a file handle
    else:
        return [m for m in _parse_transfac_file(file_descript)]


def _parse_transfac_file(transfac_fh):
    return [m for m in __parse_motif_gen(transfac_fh)]


def __parse_motif_gen(handle):

    active_motif = Motif()

    for line in handle:
        line = line.strip()

        if len(line) < 2:
            continue

        line_id, line = line[:2].upper(), line[2:].strip()

        # Spacer
        if line_id == "XX":
            continue

        # New record
        elif line_id == "//" and len(active_motif) > 0:
            yield active_motif
            active_motif = Motif()

        elif line_id == "//":
            active_motif = Motif()

        # Accession
        elif line_id == "AC":
            active_motif.accession = line

        # ID
        elif line_id == "ID":
            active_motif.motif_id = line

        # Name
        elif line_id == "NA":
            active_motif.motif_name = line

        # Alphabet
        elif line_id == "P0":
            active_motif.alphabet = line.split()

        elif line_id == "BF":
            active_motif.species = line

        # Prob
        elif line_id.isdigit():
            counts = list(map(float, line.split()[:-1]))
            active_motif.add_count_line(counts)
            total_seqs = sum(counts)
            active_motif.add_prob_line(list(map(lambda x: x / total_seqs, counts)))

    if len(active_motif) > 0:
        yield active_motif

