import subprocess
import io
import pandas as pd
import numpy as np

from inferelator_prior.motifs import chunk_motifs, homer_motif, SCAN_SCORE_COL, SCORE_PER_BASE
from inferelator_prior.motifs._motif import MotifScanner
from inferelator_prior import HOMER_EXECUTABLE_PATH

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
HOMER2_EXPAND_STR_COLS = [HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]


class HOMERScanner(MotifScanner):

    scanner_name = "HOMER"

    def _preprocess(self, min_ic=None):
        if self.motif_file is not None:
            self.motifs = homer_motif.read(self.motif_file)

        return chunk_motifs(homer_motif, self.motifs, num_workers=self.num_workers, min_ic=min_ic)

    def _postprocess(self, motif_peaks):
        motif_peaks = motif_peaks.drop_duplicates(subset=[HOMER_MOTIF, HOMER_START, HOMER_STOP, HOMER_CHROMOSOME,
                                                          HOMER_STRAND])
        return motif_peaks

    def _get_motifs(self, fasta_file, motif_file, threshold=None, parse_genomic_coord=False):
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
        loc_data.columns = HOMER2_EXPAND_STR_COLS if loc_data.shape[1] == 3 else HOMER2_EXPAND_STR_COLS + ["UNKNOWN"]
        loc_data[HOMER_START] = loc_data[HOMER_START].astype(int) + motifs[HOMER_OFFSET]

        match_width = motifs[HOMER_MATCH].str.len()

        loc_data.loc[motifs[HOMER_STRAND] == "-", HOMER_START] -= match_width.loc[motifs[HOMER_STRAND] == "-"] - 1

        loc_data[HOMER_STOP] = loc_data[HOMER_START] + motifs[HOMER_MATCH].str.len()

        motifs[[HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]] = loc_data[[HOMER_CHROMOSOME, HOMER_START, HOMER_STOP]]
        motifs.drop([HOMER_SEQ_ID, HOMER_OFFSET], inplace=True, axis=1)

        motifs[SCAN_SCORE_COL] = [self.motifs[x].score_match(y) for x, y in
                                  zip(motifs[HOMER_MOTIF], motifs[HOMER_MATCH])]
        motifs[SCORE_PER_BASE] = [np.array(self.motifs[x]._info_match(y)) for x, y in
                                  zip(motifs[HOMER_MOTIF], motifs[HOMER_MATCH])]

        return motifs
