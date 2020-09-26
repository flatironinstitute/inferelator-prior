from inferelator_prior.motifs import Motif

HOMER_MOTIF_RECORD = """\
>{consensus}\t{mname}\t{odds_score:.6f}
{pmatrix}"""

HOMER_ALPHABET = "ACGT"


def read(file_descript):

    # Parse if it's a string
    if isinstance(file_descript, str):
        with open(file_descript) as motif_fh:
            return [m for m in __parse_motif_gen(motif_fh)]

    # Parse if it's a file handle
    else:
        return [m for m in __parse_motif_gen(file_descript)]


def write(file_descript, motifs, alphabet=None, background=None, mode="w"):

    motifs = [motifs] if not isinstance(motifs, list) else motifs

    # Write if it's a string
    if isinstance(file_descript, str):
        with open(file_descript, mode=mode) as motif_fh:
            for motif in motifs:
                __write_motif(motif_fh, motif)

    # Write if it's a file handle
    else:
        for motif in motifs:
            __write_motif(file_descript, motif)


def __parse_motif_gen(handle):

    active_motif = None

    for line in handle:
        line = line.strip()

        if len(line) > 0 and line.lower().startswith(">"):
            if active_motif is not None:
                yield active_motif
            line = line.split()
            active_motif = Motif(line[1], None, list(HOMER_ALPHABET))
            active_motif.homer_odds = line[2]
        elif len(line) > 0:
            probs = line.split()
            if active_motif is not None and len(probs) == len(HOMER_ALPHABET):
                active_motif.add_prob_line(list(map(lambda x: float(x), probs)))

    if active_motif is not None:
        yield active_motif


def __write_motif(motif_fh, motif):

    if motif.alphabet is not None and "".join(motif.alphabet).upper() != HOMER_ALPHABET:
        raise ValueError("HOMER requires ACGT alphabet only")

    p_mat = "\n".join(["\t".join(map(lambda x: " {:.4f}".format(x), r)) for r in motif.probability_matrix])

    record = HOMER_MOTIF_RECORD.format(consensus=motif.consensus,
                                       mname=motif.motif_id,
                                       odds_score=motif.homer_odds,
                                       pmatrix=p_mat)
    print(record, file=motif_fh)
