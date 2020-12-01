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
    ap.add_argument("--indirect", dest="direct", help="Include indirect motifs", action='store_const',
                    const=False, default=True)
    ap.add_argument("--pwm_no_header", dest="pwm_no_header", help="PWM files have no headers", action='store_const',
                    const=True, default=False)
    ap.add_argument("--pwm_alphabet", dest="alphabet", help="PWM bases (alphabet)", metavar="ALPHABET", default=None)
    ap.add_argument("--pwm_transpose", dest="transpose", help="PWM is Bases x Positions", action='store_const',
                    const=True, default=False)

    args = ap.parse_args()

    files = []
    for mf in args.motif:
        files.extend(glob.glob(os.path.expanduser(mf)))

    pwm_to_meme(files, args.info, args.out, direct=args.direct, no_headers=args.pwm_no_header, alphabet=args.alphabet,
                transpose=args.transpose)


def pwm_to_meme(pwm_file_list, tf_info_file, output_file, direct=True, no_headers=False, alphabet=None,
                transpose=False):

    alphabet = list(alphabet) if alphabet is not None else None

    print("Parsing {x} PWM files".format(x=len(pwm_file_list)))
    motifs = read(pwm_file_list, tf_info_file, direct_only=direct, pwm_has_idx=not no_headers, pwm_alphabet=alphabet,
                  transpose=transpose)

    print("Parsed {m} motifs, writing to file {f}".format(m=len(motifs), f=output_file))
    write(output_file, list(motifs))


if __name__ == '__main__':
    main()
