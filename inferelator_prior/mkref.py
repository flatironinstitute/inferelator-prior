import argparse

from inferelator_prior.processor.star import star_mkref


def main():
    ap = argparse.ArgumentParser(description="Create a reference genome. All other arguments will be passed to STAR.")
    ap.add_argument("-f", "--fasta", dest="fasta", help="FASTA FILE(s)", nargs="+", metavar="FILE", default=None)
    ap.add_argument("-a", "--annotation", dest="annotation", help="Annotation GTF/GFF FILE", metavar="FILE",
                    default=None)
    ap.add_argument("-g", "--genome", dest="genome", help="Create standard ref genome", metavar="PATH", default=None)
    ap.add_argument("-o", "--out", dest="out", help="Output PATH", metavar="PATH", required=True)
    ap.add_argument("--cpu", dest="cpu", help="NUMBER of cores to use", metavar="PATH", type=int, default=4)

    args, star_args = ap.parse_known_args()

    if (args.fasta is None or args.annotation is None) and args.genome is None:
        print("One of (--fasta and --annotation) or --genome must be set. Not neither.")
        exit(0)
    elif (args.fasta is not None or args.annotation is not None) and args.genome is not None:
        print("One of (--fasta and --annotation) or --genome must be set. Not both.")
    elif args.genome is not None:
        star_mkref(args.out, default_genome=args.genome, cores=args.cpu, star_options=star_args)
    elif args.fasta is not None and args.annotation is not None:
        star_mkref(args.out, genome_file=args.fasta, annotation_file=args.annotation, star_options=star_args,
                   cores=args.cpu)
    else:
        raise ValueError("Switch error")


if __name__ == '__main__':
    main()
