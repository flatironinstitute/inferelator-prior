# Download the SRR file from NCBI
# TODO: make this a thing
def get_srr_file(srr_id, target_path):
    """
    Take a SRR ID string and get the SRR file for it from NCBI. Raise a ValueError if it cannot be found.

    :param srr_id: str
        NCBI SRR ID string
    :param target_path: str
        The path to put the SRR file
    :return srr_file_name: str
        The SRR file name (including path)
    """
    srr_file_name = ""
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