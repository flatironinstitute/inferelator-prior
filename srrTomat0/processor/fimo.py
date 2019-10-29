import asyncio
import os
import io
import concurrent.futures
import pandas as pd

from srrTomat0 import FIMO_EXECUTABLE_PATH
from srrTomat0.processor.meme_parser import parse_meme_file_for_record_block

BEDTOOLS_EXTRACT_SUFFIX = ".extract.fasta"
FIMO_DATA_SUFFIX = ".fimo.tsv"

FIMO_MOTIF = 'motif_id'
FIMO_MOTIF_COMMON = 'motif_alt_id'
FIMO_CHROMOSOME = 'sequence_name'
FIMO_STRAND = 'strand'
FIMO_START = 'start'
FIMO_STOP = 'stop'
FIMO_SCORE = 'p-value'


def fimo_scan(srr_ids, atac_bed_files, meme_file, genome_fasta_file, target_path, num_workers=5):
    """
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as async_pool:
        futures = [async_pool.map(_scan_fimo, sid, sfn, meme_file, genome_fasta_file, target_path)
                   for sid, sfn in zip(srr_ids, atac_bed_files)]

    futures, _ = concurrent.futures.wait(futures)

    return list(map(lambda x: x.results(), futures))


async def _scan_fimo(id_name, atac_bed_file, meme_file, genome_fasta, output_path):

    if atac_bed_file is None:
        return None

    extracted_fasta_file = await _extract_bed_sequence(id_name, atac_bed_file, genome_fasta, output_path)
    motif_data = await _get_motifs(id_name, extracted_fasta_file, meme_file, output_path)

    return motif_data


async def _extract_bed_sequence(id_name, bed_file, genome_fasta, output_path):

    output_file = os.path.join(output_path, str(id_name) + BEDTOOLS_EXTRACT_SUFFIX)

    if os.path.exists(output_file):
        return output_file

    bedtools_command = ["bedtools", "getfasta", "-fi", genome_fasta, "-bed", bed_file, "-fo", output_file]

    try:
        process = await asyncio.create_subprocess_exec(*bedtools_command)
        code = await process.wait()
    except:
        code = 1
        raise
    finally:
        if int(code) != 0:
            print("bedtools getfasta failed for {id} ({file})".format(id=id_name, file=bed_file))
            try:
                os.remove(output_file)
            except FileNotFoundError:
                pass
            return None

    return output_file


async def _get_motifs(id_name, fasta_file, meme_file, output_path):

    output_file = os.path.join(output_path, str(id_name) + FIMO_DATA_SUFFIX)

    if os.path.exists(output_file):
        return output_file

    fimo_command = [FIMO_EXECUTABLE_PATH, "--text", "--parse-genomic-coord", meme_file, fasta_file]
    code = 0

    try:
       process = await asyncio.create_subprocess_exec(*fimo_command,
                                                      stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE)
       motif_string, _ = await process.communicate()
       code = max(code, process.returncode)

    except:
        code = 1
        raise

    finally:
        if int(code) != 0:
            print("fimo motif scan failed for {id} ({file})".format(id=id_name, file=fasta_file))
            return None

    motif_data = parse_fimo_output(io.StringIO(motif_string.decode("utf-8")))
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
