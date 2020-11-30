import asyncio
import os

from inferelator_prior.processor.utils import file_path_abs
from inferelator_prior import FASTQDUMP_EXECUTABLE_PATH, PREFETCH_EXECUTABLE_PATH

PREFETCH_OPTIONS = ["--max-size", "1000000000"]

SRA_EXTENSION = ".sra"
POSSIBLE_FASTQ_EXTENSIONS = [".fastq.gz", "_1.fastq.gz", "_2.fastq.gz", "_3.fastq.gz", "_4.fastq.gz"]


# TODO: test this
def get_srr_files(srr_list, target_path, num_workers=5, prefetch_options=PREFETCH_OPTIONS, skip=False):
    """
    Take a list of SRR ID strings, download them async with num_workers concurrent jobs, and return a list of the
    paths to the SRR files that have been downloaded.
    :param srr_list: list(str)
        List of SRA IDs to acquire from NCBI
    :param target_path: str
        Target path for the SRA files
    :param num_workers: int
        Number of concurrent jobs to run
    :param prefetch_options: list(str)
        Any additional command line arguments to pass to prefetch
    :return:
    """
    sem = asyncio.Semaphore(num_workers)

    srr_file_names = list(map(lambda x: os.path.join(file_path_abs(target_path), x + SRA_EXTENSION), srr_list))
    tasks = [_get_srr(sid, sfn, sem, prefetch_options=prefetch_options, skip=skip)
             for sid, sfn in zip(srr_list, srr_file_names)]

    try:
        return asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
    except RuntimeError:
        return asyncio.new_event_loop().run_until_complete(asyncio.gather(*tasks))


# TODO: test this
async def _get_srr(srr_id, srr_file_name, semaphore, prefetch_options=PREFETCH_OPTIONS, skip=False):
    """
    Take a SRR ID string and get the SRR file for it from NCBI.

    :param srr_id: str
        NCBI SRR ID string
    :param srr_file_name: str
        The path to the SRR file (the FULL path)
    :param semaphore: asyncio.Semaphore
        Semaphore for resource utilization
    :param prefetch_options: list(str)
        Any additional command line arguments to pass to prefetch
    :return srr_file_name: str
        The SRR file name (including path)
    """
    async with semaphore:
        # If the file is already downloaded, don't do anything
        if os.path.exists(srr_file_name):
            print("{id} exists in file {file}".format(id=srr_id, file=srr_file_name))
            return srr_file_name

        if skip:
            return srr_file_name

        prefetch_call = [PREFETCH_EXECUTABLE_PATH, srr_id, "-o", srr_file_name, *prefetch_options]
        process = await asyncio.create_subprocess_exec(*prefetch_call)
        code = await process.wait()

        if int(code) != 0:
            print("NCBI Prefetch failed for {id} ({file})".format(id=srr_id, file=srr_file_name))
            print(" ".join(prefetch_call))
            return None

        return srr_file_name


# TODO: test this
def unpack_srr_files(srr_ids, srr_file_names, target_path, num_workers=5, skip=False):
    """
    Take an SRR file and unpack it into a set of FASTQ files

    :param srr_ids: list(str)
        NCBI SRR ID string
    :param srr_file_names: list(str)
        The complete path to the SRR file
    :param target_path: str
        The path to put the FASTQ file(s)
    :param num_workers: int
        Number of concurrent jobs to run
    :return fastq_file_names: list
        A list of complete FASTQ file names that were unpacked from the SRR file (including path)
    """

    sem = asyncio.Semaphore(num_workers)

    tasks = [_unpack_srr(sid, sfn, target_path, sem, skip=skip) for sid, sfn in zip(srr_ids, srr_file_names)]
    return asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))


# TODO: test this
async def _unpack_srr(srr_id, srr_file_name, target_path, semaphore, skip=False):
    """

    :param srr_id: str
        NCBI SRR ID string
    :param srr_file_name: str
        The complete path to the SRR file
    :param target_path: str
        The path to put the FASTQ file(s)
    :param semaphore: asyncio.Semaphore
        Semaphore for resource utilization
    :return:
    """
    async with semaphore:

        if srr_file_name is None:
            return [None]

        # Check and see if this has already been done
        output_file_names = list(map(lambda x: os.path.join(file_path_abs(target_path), srr_id + x),
                                     POSSIBLE_FASTQ_EXTENSIONS))
        files_created = check_list_of_files_exist(output_file_names)

        if skip:
            return files_created

        # If the file is already unpacked, don't do anything
        if len(files_created) > 0:
            print("{id} exists in path {path} ({files})".format(id=srr_id, path=target_path,
                                                                files=" ".join(files_created)))
            return files_created

        # Build a fastq-dump call and execute it
        fastq_dump_call = [FASTQDUMP_EXECUTABLE_PATH, "--gzip", "--split-files", "--outdir", target_path,
                           srr_file_name]

        # Run fastq-dump and get the files that were created from it
        return_code = 0
        try:
            process = await asyncio.create_subprocess_exec(*fastq_dump_call)
            return_code = await process.wait()
            file_output = check_list_of_files_exist(output_file_names)
        except:
            return_code = 1
            file_output = [None]
            raise
        finally:
            # If the fastq-dump failed, clean up the files associated with it and then move on
            if int(return_code) != 0:
                print("NCBI fastq-dump failed for {id} ({file})".format(id=srr_id, file=srr_file_name))
                print(" ".join(fastq_dump_call))
                files_created = check_list_of_files_exist(output_file_names)
                for f in files_created:
                    try:
                        os.remove(f)
                    except FileNotFoundError:
                        pass
                file_output = [None]

        # Find out which read files were created by looking into the output folder
        return file_output


def check_list_of_files_exist(file_list):
    """
    Check a list of file names and return subset of the list that exists (or an empty list if none exist)
    :param file_list: list(str)
        List of file names
    :return existing_file_list: list(str)
        List of files that exist
    """

    existing_file_list = []

    for file_name in file_list:
        if os.path.exists(file_path_abs(file_name)):
            existing_file_list.append(file_name)

    return existing_file_list
