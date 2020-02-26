from srrTomat0.motifs.pwm import read
from srrTomat0.motifs.meme import write

import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser(description="Parse naked PWM files into a MEME file")
    ap.add_argument("-m", "--motif", dest="motif", help="Motif PWM files", metavar="PATH", required=True, nargs="+")
    ap.add_argument("-i", "--info", dest="info", help="Motif Info File", metavar="PATH", required=True)
    ap.add_argument("-o", "--out", dest="out", help="Output FILE", metavar="FILE", required=True)

    args = ap.parse_args()

    files = []
    for mf in args.motif:
        files.extend(glob.glob(os.path.expanduser(mf)))

    pwm_to_meme(files, args.info, args.out)


def pwm_to_meme(pwm_file_list, tf_info_file, output_file):

    motifs = read(pwm_file_list, tf_info_file)

    motifs_name = {}

    for m in motifs:
        if m.motif_name in motifs_name and m.information_content > motifs_name[m.motif_name].information_content:
            motifs_name[m.motif_name] = m
        elif m.motif_name not in motifs_name:
            motifs_name[m.motif_name] = m
        else:
            pass

    write(output_file, list(motifs_name.values()))


if __name__ == '__main__':
    main()
