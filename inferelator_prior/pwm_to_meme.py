from inferelator_prior.motifs.pwm import read
from inferelator_prior.motifs.meme import write

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

    print("Parsing {x} PWM files".format(x=len(pwm_file_list)))
    motifs = read(pwm_file_list, tf_info_file, direct_only=True)

    print("Parsed {m} motifs, writing to file {f}".format(m=len(motifs), f=output_file))
    write(output_file, list(motifs))


if __name__ == '__main__':
    main()
