import argparse

from srrTomat0.processor.star import star_mkref


def main():
    ap = argparse.ArgumentParser(description="Create a reference genome")
    ap.add_argument("-f", "--fasta", dest="fasta", help="FASTA FILE(s)", nargs="+", metavar="FILE", default=None)
    ap.add_argument("-a", "--annotation", dest="annotation", help="Annotation GTF/GFF FILE", metavar="FILE",
                    default=None)
    ap.add_argument("-g", "--genome", dest="genome", help="Create standard ref genome", metavar="PATH", default=None)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    args = ap.parse_args()

    if (args.fasta is None or args.annotation is None) and args.genome is None:
        print("One of (--fasta and --annotation) or --genome must be set. Not neither.")
        exit(0)
    elif (args.fasta is not None or args.annotation is not None) and args.genome is not None:
        print("One of (--fasta and --annotation) or --genome must be set. Not both.")
    elif args.genome is not None:
        star_mkref(args.out, default_genome=args.genome)
    elif args.fasta is not None and args.annotation is not None:
        star_mkref(args.out, genome_file=args.fasta, annotation_file=args.annotation)
    else:
        raise ValueError("Switch error")


if __name__ == '__main__':
    main()
