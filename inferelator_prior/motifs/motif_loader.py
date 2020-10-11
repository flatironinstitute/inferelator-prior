from inferelator_prior.motifs import motifs_to_dataframe


def load_motif_file(motif_file, motif_format):

    motif_format = motif_format.lower()

    print("Loading motifs from file ({f})".format(f=motif_file))
    if motif_format == "meme":
        from inferelator_prior.motifs.meme import read
    elif motif_format == "transfac":
        from inferelator_prior.motifs.transfac import read
    elif motif_format == "homer":
        from inferelator_prior.motifs.homer_motif import read
    else:
        raise ValueError("motif_format must be 'meme', 'homer', or 'transfac'")

    motifs = read(motif_file)
    motif_information = motifs_to_dataframe(motifs)

    print("\t{n} motifs loaded".format(n=len(motif_information)))

    return motifs, motif_information
