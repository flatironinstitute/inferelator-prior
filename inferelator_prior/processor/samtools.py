import asyncio
import os

from inferelator_prior import SAMTOOLS_EXECUTABLE_PATH

BAM_EXTENSION = ".bam"


def sam_sort(srr_ids, sam_files, target_path, min_quality=None, num_workers=5):
    """
    Sort (and filter) SAM files into BAM files

    :param srr_ids: list(str)
        List of SRA IDs to acquire from NCBI
    :param sam_files: list(str)
        List of SAM file paths
    :param target_path: str
        Target path for the SRA files
    :param min_quality: int
        If set, filter reads for MINQ
    :param num_workers: int
        Number of concurrent jobs to run
    :return:
    """

    sem = asyncio.Semaphore(num_workers)

    tasks = [_process_sam(sid, sfn, target_path, sem, min_quality=min_quality) for sid, sfn in zip(srr_ids, sam_files)]

    return asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))


async def _process_sam(srr_id, sam_file, target_path, semaphore, min_quality=None):
    """
    Sort (and filter) SAM file into BAM file
    :param srr_id: str
        NCBI SRR ID string
    :param sam_file:
        The complete path to the SAM file
    :param target_path: str
        The path to put the BAM file
    :param semaphore: asyncio.Semaphore
        Semaphore for resource utilization
    :param min_quality: int
        Minimum alignment quality score to include (None disables filter)
    :return bam_file_name: str
        Path to the created BAM file name
    """
    async with semaphore:

        if sam_file is None:
            return None

        bam_file_name = os.path.join(target_path, srr_id + BAM_EXTENSION)

        samtools_sort_call = [SAMTOOLS_EXECUTABLE_PATH]
        sort_cmd = ["sort", "-o", bam_file_name]

        # If min_quality is set, pipe in a view with a -q flag set
        if min_quality is not None:
            samtools_sort_call.extend(["view", "-q", str(min_quality), sam_file, "|",
                                       SAMTOOLS_EXECUTABLE_PATH, *sort_cmd, "-"])
        else:
            samtools_sort_call.extend([*sort_cmd, sam_file])

        # Create a sorted BAM file
        try:
            print(" ".join(samtools_sort_call))
            process = await asyncio.create_subprocess_exec(*samtools_sort_call)
            code = await process.wait()
        except:
            code = 1
            raise
        finally:
            if int(code) != 0:
                print("samtools sort failed for {id} ({file})".format(id=srr_id, file=bam_file_name))
                try:
                    os.remove(bam_file_name)
                except FileNotFoundError:
                    pass
                return None

        return bam_file_name
