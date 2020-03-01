import os
import pandas as pd
import tempfile
import pathos
import subprocess
import io
import pandas as pd

from srrTomat0.motifs import MotifScanner, chunk_motifs, homer_motif
from srrTomat0.processor.bedtools import extract_bed_sequence
from srrTomat0 import HOMER_EXECUTABLE_PATH


HOMER_DATA_SUFFIX = ".homer.tsv"

HOMER_SEQ_ID = 'seqid'
HOMER_OFFSET = 'offset'
HOMER_MATCH = 'match'
HOMER_MOTIF = 'motif_id'
HOMER_STRAND = 'strand'
HOMER_SCORE = 'score'
HOMER_CHROMOSOME = 'sequence_name'
HOMER_START = 'start'
HOMER_STOP = 'stop'

HOMER2_FIND_COLS = [HOMER_SEQ_ID, HOMER_OFFSET, HOMER_MATCH, HOMER_MOTIF, HOMER_STRAND, HOMER_SCORE]


class HOMERScanner(MotifScanner):

    def _preprocess(self, min_ic=None):
        return chunk_motifs(homer_motif, motif_file=self.motif_file, motifs=self.motifs, num_workers=self.num_workers,
                            min_ic=min_ic)

    def _get_motifs(self, fasta_file, motif_file):
        homer_command = [HOMER_EXECUTABLE_PATH, "find", "-i", fasta_file, "-m", motif_file, "-offset", str(0)]
        proc = subprocess.run(homer_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if int(proc.returncode) != 0:
            print("HOMER motif scan failed for {meme}, {fa} (cmd)".format(meme=motif_file,
                                                                         fa=fasta_file,
                                                                         cmd=" ".join(homer_command)))

        return self._parse_output(io.StringIO(proc.stdout.decode("utf-8")))

    def _parse_output(self, output_handle):
        motifs = pd.read_csv(output_handle, sep="\t", index_col=None, names=HOMER2_FIND_COLS)

        loc_data = motifs[HOMER_SEQ_ID].str.split(r"[\:\-]", expand=True)
        loc_data.columns = [HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]
        loc_data[HOMER_START] = loc_data[HOMER_START].astype(int) + motifs[HOMER_OFFSET]
        loc_data[HOMER_STOP] = loc_data[HOMER_START] + motifs[HOMER_MATCH].str.len()

        motifs[[HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]] = loc_data
        motifs[HOMER_SEQ_ID] = None
        motifs[HOMER_OFFSET] = None
        return motifs


def homer_scan(atac_bed_file, genome_fasta_file, motif_file=None, motifs=None, target_path=None, num_workers=4,
               min_ic=None):
    """
    """

    if (motif_file is None and motifs is None) or (motif_file is not None and motifs is not None):
        raise ValueError("One of meme_file or motifs must be passed")

    fasta_output = tempfile.gettempdir() if target_path is None else os.path.abspath(os.path.expanduser(target_path))

    # Extract interesting sequences to a temp fasta file
    extracted_fasta_file = extract_bed_sequence(atac_bed_file, genome_fasta_file, fasta_output)

    # Preprocess motifs into a list of temp chunk files
    motif_files = chunk_motifs(homer_motif, motif_file=motif_file, motifs=motifs, num_workers=num_workers,
                               min_ic=min_ic)

    try:
        # If the number of workers is 1, run fimo directly
        if num_workers == 1:
            assert len(motif_files) == 1
            return _get_motifs(extracted_fasta_file, motif_files[0], output_path=target_path)

        # Otherwise parallelize with a process pool (pathos because dill will do local functions)
        else:
            # Convenience local function
            def _get_chunk_motifs(chunk_file):
                return _get_motifs(extracted_fasta_file, chunk_file)

            with pathos.multiprocessing.Pool(num_workers) as pool:
                motif_data = [data for data in pool.imap(_get_chunk_motifs, motif_files)]
                motif_data = pd.concat(motif_data)

        return motif_data

    # Clean up the temporary files
    finally:
        os.remove(extracted_fasta_file)

        for file in motif_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass


def _get_motifs(fasta_file, motif_file, output_path=None):

    if output_path is not None:
        output_file = os.path.join(output_path, fasta_file + HOMER_DATA_SUFFIX)

        if os.path.exists(output_file):
            return pd.read_csv(output_file, sep="\t", index_col=None)
    else:
        output_file = None

    homer_command = [HOMER_EXECUTABLE_PATH, "find", "-i", fasta_file, "-m", motif_file, "-offset", str(0)]
    proc = subprocess.run(homer_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    if int(proc.returncode) != 0:
        print("fimo motif scan failed for {meme}, {fa} (cmd)".format(meme=motif_file,
                                                                     fa=fasta_file,
                                                                     cmd=" ".join(homer_command)))

    motif_data = parse_homer_output(io.StringIO(proc.stdout.decode("utf-8")))

    if output_file is not None:
        motif_data.to_csv(output_file, sep="\t", index=False)

    return motif_data


def parse_homer_output(buffer):

    motifs = pd.read_csv(buffer, sep="\t", index_col=None, names=HOMER2_FIND_COLS)

    loc_data = motifs[HOMER_SEQ_ID].str.split(r"[\:\-]", expand=True)
    loc_data.columns = [HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]
    loc_data[HOMER_START] = loc_data[HOMER_START].astype(int) + motifs[HOMER_OFFSET]
    loc_data[HOMER_STOP] = loc_data[HOMER_START] + motifs[HOMER_MATCH].str.len()

    motifs[[HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]] = loc_data
    motifs[HOMER_SEQ_ID] = None
    motifs[HOMER_OFFSET] = None
    return motifs

