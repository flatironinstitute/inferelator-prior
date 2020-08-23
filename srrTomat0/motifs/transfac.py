from srrTomat0.motifs import Motif

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

    active_motif = None
    active_ac, active_id, active_species, active_name = None, None, None, []

    for line in handle:
        line = line.strip().lower()

        # Spacer
        if line.startswith("XX"):
            continue

        # New record
        elif line.startswith("//") and active_motif is not None:
            yield active_motif
            active_ac, active_id, active_alphabet, active_species = None, None, None, []

        # Accession
        elif line.startswith("AC"):
            active_ac = line[2:].strip()

        # ID
        elif line.startswith("ID"):
            active_id = line[2:].strip()

        # Name
        elif line.startswith("NA"):
            active_name = line[2:].strip()

        # Alphabet
        elif line.startswith("P0"):
            active_motif = Motif(active_ac, active_name, line[2:].strip().split())

        # Prob
        elif line[:2].isdigit():
            probs = line[2:].strip().split()[:-1]
            total_seqs = sum(line)
            active_motif.add_prob_line(list(map(lambda x: float(x) / total_seqs, probs)))

    if active_motif is not None:
        yield active_motif
