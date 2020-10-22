from inferelator_prior.motifs import Motif

import numpy as np
import pandas as pd

MEME4_HEADER = """\
MEME version 4

ALPHABET= {alphabet}

strands: {strands}

Background letter frequencies:
{bkgd}
"""

MEME4_RECORD = """\
MOTIF {motif_id} {motif_name}

letter-probability matrix: alength= {alen} w= {w}
{pmatrix}

URL {url}
"""


def read(file_descript):

    # Parse if it's a string
    if isinstance(file_descript, str):
        with open(file_descript) as motif_fh:
            return [m for m in __parse_meme_file(motif_fh)]

    # Parse if it's a file handle
    else:
        return [m for m in __parse_meme_file(file_descript)]


def write(file_descript, motifs, alphabet=None, background=None, mode="w"):

    motifs = [motifs] if not isinstance(motifs, (list, tuple, pd.Series)) else motifs
    alphabet = alphabet if alphabet is not None else motifs[0].alphabet
    background = np.array([[1 / len(alphabet)] * len(alphabet)]) if background is None else background

    def _write_file(fh):
        __write_header(fh, alphabet, background)
        for motif in motifs:
            __write_motif(fh, motif)

    # Write if it's a string
    if isinstance(file_descript, str):
        with open(file_descript, mode=mode) as motif_fh:
            _write_file(motif_fh)

    # Write if it's a file handle
    else:
        _write_file(file_descript)


def __parse_meme_file(meme_fh):
    alph = __parse_alphabet(meme_fh)

    neg_strand, pos_strand = __parse_strand(meme_fh, strict=False)

    bkgd = __parse_background(meme_fh, strict=False)
    bkgd = np.array([[1 / len(alph)] * len(alph)]) if bkgd is None else np.array([[bkgd[a] for a in alph]])

    return [m for m in __parse_motif_gen(meme_fh, alph, bkgd)]


def __parse_alphabet(handle, strict=True):

    for line in handle:
        if line.strip().lower().startswith("alphabet"):
            handle.seek(0)
            return list(line.strip().split()[-1])

    if strict:
        raise MEMEDatabaseError("Unable to locate `ALPHABET =` line")


def __parse_strand(handle, strict=True):

    for line in handle:
        if line.strip().lower().startswith("strands"):
            handle.seek(0)
            strands = "".join(line.strip().split()[-2:])
            return "-" in strands, "+" in strands

    if strict:
        raise MEMEDatabaseError("Unable to locate `ALPHABET =` line")
    else:
        handle.seek(0)
        return True, True


def __parse_background(handle, strict=True):

    find_flag = False

    for line in handle:
        line = line.strip()
        if line.lower().startswith("background"):
            find_flag = True
            continue
        if len(line) > 0 and find_flag:
            probs = line.split()

            if len(probs) % 2 != 0:
                raise MEMEDatabaseError("Background probabilities do not parse correctly")

            handle.seek(0)
            return {a: float(b) for a, b in zip(probs[::2], probs[1::2])}

    if strict:
        raise MEMEDatabaseError("Unable to locate background probabilities")
    else:
        return None


def __parse_motif_gen(handle, alphabet, background):

    active_motif = None

    for line in handle:
        line = line.strip()

        if active_motif is None and line.lower().startswith("motif"):
            line = line.split()
            active_motif = Motif(line[1], line[2] if len(line) > 2 else None, alphabet, background)
            continue
        elif active_motif is not None and line.lower().startswith("motif"):
            yield active_motif
            line = line.split()
            active_motif = Motif(line[1], line[2] if len(line) > 2 else None, alphabet, background)
            continue

        if line.lower().startswith("letter-probability") or len(line) == 0:
            continue

        if line.lower().startswith("url") and active_motif is not None:
            active_motif.motif_url = line.split()[-1].strip()
            continue

        probs = line.split()
        if active_motif is not None and len(probs) == len(alphabet):
            active_motif.add_prob_line(list(map(lambda x: float(x), probs)))

    if active_motif is not None:
        yield active_motif


def __write_header(handle, alphabet, bkgd, pos_strand=True, neg_strand=True):

    strands = ["-"] if neg_strand and not pos_strand else ['+'] if pos_strand and not neg_strand else ["-", "+"]
    bkgd = ["{} {:.5f}".format(a, b) for a, b in zip(alphabet, bkgd.flatten().tolist())]

    meme4_header = MEME4_HEADER.format(alphabet="".join(alphabet),
                                       strands=" ".join(strands),
                                       bkgd=" ".join(bkgd))

    print(meme4_header, file=handle)


def __write_motif(handle, motif):

    p_mat = "\n".join(["\t".join(map(lambda x: "  {:.6f}".format(x), r)) for r in motif.probability_matrix])

    meme4_record = MEME4_RECORD.format(motif_id=motif.motif_id if motif.motif_id is not None else "",
                                       motif_name=motif.motif_name if motif.motif_name is not None else "",
                                       alen=motif.alphabet_len,
                                       w=len(motif),
                                       pmatrix=p_mat,
                                       url=motif.motif_url if motif.motif_url is not None else "")

    print(meme4_record, file=handle)


class MEMEDatabaseError(ValueError):
    pass
