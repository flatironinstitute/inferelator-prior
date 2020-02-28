import os
import io
import subprocess
import pathos
import pandas as pd
import tempfile
import math
import itertools

from srrTomat0 import FIMO_EXECUTABLE_PATH
from srrTomat0.motifs import meme

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"
FIMO_DATA_SUFFIX = ".fimo.tsv"

FIMO_MOTIF = 'motif_id'
FIMO_MOTIF_COMMON = 'motif_alt_id'
FIMO_CHROMOSOME = 'sequence_name'
FIMO_STRAND = 'strand'
FIMO_START = 'start'
FIMO_STOP = 'stop'
FIMO_SCORE = 'p-value'


def fimo_scan(atac_bed_file, genome_fasta_file, meme_file=None, motifs=None, target_path=None, num_workers=4,
              min_ic=None):
    """
    """

    if (meme_file is None and motifs is None) or (meme_file is not None and motifs is not None):
        raise ValueError("One of meme_file or motifs must be passed")

    fasta_output = tempfile.gettempdir() if target_path is None else os.path.abspath(os.path.expanduser(target_path))

    # Extract interesting sequences to a temp fasta file
    extracted_fasta_file = _extract_bed_sequence(atac_bed_file, genome_fasta_file, fasta_output)

    # Preprocess motifs into a list of temp chunk files
    meme_files = _chunk_motifs(meme_file=meme_file, motifs=motifs, num_workers=num_workers, min_ic=min_ic)

    try:

        # If the number of workers is 1, run fimo directly
        if num_workers == 1:
            assert len(meme_files) == 1
            return _get_motifs(extracted_fasta_file, meme_files[0], output_path=target_path)

        # Otherwise parallelize with a process pool (pathos because dill will do local functions)
        else:
            # Convenience local function
            def _get_chunk_motifs(chunk_file):
                return _get_motifs(extracted_fasta_file, chunk_file)

            with pathos.multiprocessing.Pool(num_workers) as pool:
                motif_data = [data for data in pool.imap(_get_chunk_motifs, meme_files)]
                motif_data = pd.concat(motif_data)

        return motif_data

    # Clean up the temporary files
    finally:
        os.remove(extracted_fasta_file)

        for file in meme_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass


def _extract_bed_sequence(bed_file, genome_fasta, output_path):

    output_file = os.path.join(output_path, genome_fasta + BEDTOOLS_EXTRACT_SUFFIX)

    if os.path.exists(output_file):
        return output_file

    bedtools_command = ["bedtools", "getfasta", "-fi", genome_fasta, "-bed", bed_file, "-fo", output_file]
    proc = subprocess.run(bedtools_command)

    if int(proc.returncode) != 0:
        print("bedtools getfasta failed for {file} ({cmd})".format(file=bed_file, cmd=" ".join(bedtools_command)))
        try:
            os.remove(output_file)
        except FileNotFoundError:
            pass
        return None

    return output_file


def _chunk_motifs(meme_file=None, motifs=None, num_workers=4, min_ic=None):
    """
    Break a motif file up into chunks
    :param meme_file: MEME file name; pass either meme_file or motifs
    :type meme_file: str, None
    :param motifs: Motif object list; pass either meme_file or motifs
    :type motifs: list(Motif), None
    :param num_workers: number of chunks to make
    :type num_workers: int
    :param min_ic: set an information content minimum on motifs to include if this is not None
    :type min_ic: float
    :return:
    """

    temp_dir = tempfile.gettempdir()

    if (meme_file is None and motifs is None) or (meme_file is not None and motifs is not None):
        raise ValueError("One of meme_file or motifs must be passed")
    if meme_file is not None:
        motifs = meme.read(meme_file)

    if min_ic is not None:
        motifs = list(itertools.compress(motifs, [m.information_content >= min_ic for m in motifs]))

    if num_workers == 1:
        file_name = os.path.join(temp_dir, "chunk1.meme")
        meme.write(file_name, motifs)
        return [file_name]

    chunk_size = math.ceil(len(motifs) / num_workers)

    files = []

    for i in range(num_workers):
        file_name = os.path.join(temp_dir, "chunk" + str(i) + ".meme")
        meme.write(file_name, motifs[i * chunk_size:min((i+1) * chunk_size, len(motifs))])
        files.append(file_name)

    return files


def _get_motifs(fasta_file, meme_file, output_path=None):

    if output_path is not None:
        output_file = os.path.join(output_path, fasta_file + FIMO_DATA_SUFFIX)

        if os.path.exists(output_file):
            return pd.read_csv(output_file, sep="\t", index_col=None)
    else:
        output_file = None

    fimo_command = [FIMO_EXECUTABLE_PATH, "--skip-matched-sequence", "--parse-genomic-coord", meme_file, fasta_file]
    proc = subprocess.run(fimo_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    if int(proc.returncode) != 0:
        print("fimo motif scan failed for {meme}, {fa} (cmd)".format(meme=meme_file,
                                                                     fa=fasta_file,
                                                                     cmd=" ".join(fimo_command)))

    motif_data = parse_fimo_output(io.StringIO(proc.stdout.decode("utf-8")))

    if output_file is not None:
        motif_data.to_csv(output_file, sep="\t", index=False)

    return motif_data


def parse_fimo_output(fimo_output_file):
    """

    :param fimo_output_file: str
        FIMO output file path or FIMO output in a StringIO object
    :return:
    """
    motifs = pd.read_csv(fimo_output_file, sep="\t", index_col=None)
    motifs.dropna(subset=[FIMO_START, FIMO_STOP], inplace=True, how='any')
    motifs[FIMO_START], motifs[FIMO_STOP] = motifs[FIMO_START].astype(int), motifs[FIMO_STOP].astype(int)
    return motifs
