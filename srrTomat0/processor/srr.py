import os
import asyncio

from srrTomat0.processor.utils import file_path_abs

NCBI_PREFETCH_EXECUTABLE = "prefetch"


def get_srr_files(srr_list, target_path, num_workers=5):
    """
    Take a list of SRR ID strings, download them async with num_workers concurrent jobs, and return a list of the
    paths to the SRR files that have been downloaded.
    :param srr_list: list(str)
        List of SRA IDs to acquire from NCBI
    :param target_path: str
        Target path for the SRA files
    :param num_workers: int
        Number of concurrent jobs to run
    :return:
    """
    sem = asyncio.Semaphore(num_workers)

    srr_file_names = list(map(lambda x: os.path.join(file_path_abs(target_path), x + ".sra"), srr_list))

    async def gather_results():
        await asyncio.gather(*[get_srr_file(sid, sfn, sem) for sid, sfn in zip(srr_list, srr_file_names)])

    asyncio.get_event_loop().run_until_complete(gather_results())

    return srr_file_names


async def get_srr_file(srr_id, srr_file_name, semaphore):
    """
    Take a SRR ID string and get the SRR file for it from NCBI.

    :param srr_id: str
        NCBI SRR ID string
    :param srr_file_name: str
        The path to the SRR file (the FULL path)
    :param semaphore: asyncio.Semaphore
        Semaphore for resource utilization
    :return srr_file_name: str
        The SRR file name (including path)
    """
    async with semaphore:
        # If the file is already downloaded, don't do anything
        if os.path.exists(srr_file_name):
            print("{id} exists in file {file}".format(id=srr_id, file=srr_file_name))
            return srr_file_name

        prefetch_call = [NCBI_PREFETCH_EXECUTABLE, srr_id, "-o", srr_file_name]
        print(" ".join(prefetch_call))
        process = await asyncio.create_subprocess_exec(*prefetch_call)
        code = await process.wait()

        if int(code) != 0:
            raise ValueError("NCBI Prefetch failed for {id} ({file})".format(id=srr_id, file=srr_file_name))

        return srr_file_name


# Unpack the SRR file to a fastQ file
# TODO: make this a thing
def unpack_srr_file(srr_id, srr_file_name, target_path):
    """
    Take an SRR file and unpack it into a set of FASTQ files

    :param srr_id: str
        NCBI SRR ID string
    :param srr_file_name: str
        The complete path to the SRR file
    :param target_path: str
        The path to put the FASTQ file(s)
    :return fastq_file_names: list
        A list of complete FASTQ file names that were unpacked from the SRR file (including path)
    """
    fastq_file_names = []
    return fastq_file_names