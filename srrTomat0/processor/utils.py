import shutil
import os
import urllib.parse
import urllib.request


def get_file_from_url(file_url, file_name_local=None):
    """
    Download a file from a url to a local file
    :param file_url:
    :param file_name_local:
    :return:
    """

    if file_name_local is None:
        file_name_local = file_path_abs(urllib.parse.urlsplit(file_url).path.split("/")[-1])

    print("Downloading {url} to {file}".format(url=file_url, file=file_name_local))

    with urllib.request.urlopen(file_url) as remote_handle, open(file_name_local, mode="wb") as local_handle:
        shutil.copyfileobj(remote_handle, local_handle)

    return file_name_local


def file_path_abs(file_path):
    """
    Convert a file path to a safe absolute path
    :param file_path: str
    :return: str
    """
    return os.path.abspath(os.path.expanduser(file_path))

