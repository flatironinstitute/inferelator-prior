import io
import subprocess
import pandas as pd
import pandas.errors as pde

from srrTomat0 import FIMO_EXECUTABLE_PATH
from srrTomat0.motifs import meme, chunk_motifs, SCAN_SCORE_COL
from srrTomat0.motifs._motif import __MotifScanner

FIMO_DATA_SUFFIX = ".fimo.tsv"

FIMO_MOTIF = 'motif_id'
FIMO_MOTIF_COMMON = 'motif_alt_id'
FIMO_CHROMOSOME = 'sequence_name'
FIMO_STRAND = 'strand'
FIMO_START = 'start'
FIMO_STOP = 'stop'
FIMO_SCORE = 'p-value'
FIMO_SEQUENCE = 'matched_sequence'

FIMO_COMMAND = [FIMO_EXECUTABLE_PATH, "--text", "--parse-genomic-coord"]


class FIMOScanner(__MotifScanner):

    def _preprocess(self, min_ic=None):
        if self.motif_file is not None:
            self.motifs = meme.read(self.motif_file)

        return chunk_motifs(meme, self.motifs, num_workers=self.num_workers, min_ic=min_ic)

    def _postprocess(self, motif_peaks):
        motif_peaks = motif_peaks.drop_duplicates(subset=[FIMO_MOTIF, FIMO_START, FIMO_STOP, FIMO_CHROMOSOME])
        return motif_peaks

    def _get_motifs(self, fasta_file, motif_file, threshold=None):
        if threshold is None:
            fimo_command = FIMO_COMMAND + [motif_file, fasta_file]
        else:
            fimo_command = FIMO_COMMAND + ["--thresh", str(threshold)] + [motif_file, fasta_file]

        proc = subprocess.run(fimo_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if int(proc.returncode) != 0:
            print("fimo motif scan failed for {meme}, {fa} ({cmd})".format(meme=motif_file,
                                                                           fa=fasta_file,
                                                                           cmd=" ".join(fimo_command)))

        return self._parse_output(io.StringIO(proc.stdout.decode("utf-8")))

    def _parse_output(self, output_handle):
        try:
            motifs = pd.read_csv(output_handle, sep="\t", index_col=None)
            motifs.dropna(subset=[FIMO_START, FIMO_STOP], inplace=True, how='any')
            motifs[FIMO_START], motifs[FIMO_STOP] = motifs[FIMO_START].astype(int), motifs[FIMO_STOP].astype(int)

            motifs[SCAN_SCORE_COL] = [self.motifs[x].score_match(y) for x, y in
                                      zip(motifs[FIMO_MOTIF], motifs[FIMO_SEQUENCE])]

            return motifs
        except pde.EmptyDataError:
            return None
