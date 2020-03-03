import io
import subprocess
import pandas as pd
import pandas.errors as pde

from srrTomat0 import FIMO_EXECUTABLE_PATH
from srrTomat0.motifs import MotifScanner, meme, chunk_motifs

FIMO_DATA_SUFFIX = ".fimo.tsv"

FIMO_MOTIF = 'motif_id'
FIMO_MOTIF_COMMON = 'motif_alt_id'
FIMO_CHROMOSOME = 'sequence_name'
FIMO_STRAND = 'strand'
FIMO_START = 'start'
FIMO_STOP = 'stop'
FIMO_SCORE = 'p-value'

FIMO_COMMAND = [FIMO_EXECUTABLE_PATH, "--skip-matched-sequence", "--parse-genomic-coord"]


class FIMOScanner(MotifScanner):

    def _preprocess(self, min_ic=None):
        return chunk_motifs(meme, motif_file=self.motif_file, motifs=self.motifs, num_workers=self.num_workers,
                            min_ic=min_ic)

    def _get_motifs(self, fasta_file, motif_file):
        fimo_command = FIMO_COMMAND + [motif_file, fasta_file]
        proc = subprocess.run(fimo_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if int(proc.returncode) != 0:
            print("fimo motif scan failed for {meme}, {fa} ({cmd})".format(meme=motif_file,
                                                                           fa=fasta_file,
                                                                           cmd=" ".join(fimo_command)))
            print(proc.stdout.decode("utf-8"))

        motif_data = self._parse_output(io.StringIO(proc.stdout.decode("utf-8")))
        return motif_data

    def _parse_output(self, output_handle):
        try:
            motifs = pd.read_csv(output_handle, sep="\t", index_col=None)
            motifs.dropna(subset=[FIMO_START, FIMO_STOP], inplace=True, how='any')
            motifs[FIMO_START], motifs[FIMO_STOP] = motifs[FIMO_START].astype(int), motifs[FIMO_STOP].astype(int)
            return motifs
        except pde.EmptyDataError:
            return None
