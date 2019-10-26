import copy


MEME_VERSION_HEADER = "MEME", "version"
MEME_VERSION = "version"
MEME_MOTIF = "MOTIF"


def parse_meme_file_for_record_block(file_name):

    meme_record = []
    meme_header = []
    open_record = None

    with open(file_name, mode="r") as meme_fh:
        for line in meme_fh:
            line = line.rstrip()
            if line[0:len(MEME_MOTIF)].lower() == MEME_MOTIF.lower():
                current_name = open_record
                open_record, _ = _parse_meme_name_line(line)
                if current_name is None:
                    meme_header = copy.copy(meme_record)
                else:
                    yield current_name, meme_record, meme_header
                meme_record = [line]
            else:
                meme_record.append(line)

    yield open_record, meme_record, meme_header
    raise StopIteration


def _parse_meme_name_line(name_line):
    name_arr = list(map(lambda x: x.strip(), name_line.split()))
    if len(name_arr) == 1:
        return "", ""
    elif len(name_arr) == 2:
        return name_arr[1], ""
    else:
        return name_arr[1], name_arr[2]


