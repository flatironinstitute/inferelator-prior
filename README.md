# inferelator-prior
[![PyPI version](https://badge.fury.io/py/inferelator-prior.svg)](https://badge.fury.io/py/inferelator-prior)
[![CI](https://github.com/flatironinstitute/inferelator-prior/actions/workflows/python-package.yml/badge.svg)](https://github.com/flatironinstitute/inferelator-prior/actions/workflows/python-package.yml/)
[![codecov](https://codecov.io/gh/flatironinstitute/inferelator-prior/branch/release/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/inferelator-prior)


This is a set of pipelines to create expression, dynamic response, and prior matrices for network inference. 
They are designed to create data that is compatible with the [inferelator](https://github.com/flatironinstitute/inferelator) package. 

### Usage

    python -m inferelator_prior.network_from_motifs
    usage: network_from_motifs.py --motif motif_PWM_file.meme
                                  -f genome_fasta_file.fasta
                                  -g genome_annotation_file.gtf
                                  -o ~/output/path/prefix
                                  --species {yeast,fly,mouse,human}
                                  -b constraning_bed_file.bed
                                  --cpu num_cores
                                  --genes gene_list.txt
                                  --tfs tf_list.txt
                                  
This requires a motif PWM database (`-m PATH`), 
a genome to search (both sequence as a FASTA `-f PATH` and annotations `-g PATH`),
and an output prefix for several files (`-o PATH`).
In addition, default settings for a specific species can be set with (`--species`).
A BED file can be provided (`-b PATH`) based on some constraining experiment to restrict searching to 
specific genomic areas.
This will use multiple cores to search for motifs and process the resulting data.
By default, all available processors will be used, but this can be overridden with `--cpu N`.
A list of genes (on e per line in a text file) can be provided to `--genes` 
and a list of tfs (one per line in a text file) can be provided to `--tfs`.
A network will be built for only these genes or TFs.

The output from this script is an unsigned connectivity matrix (0, 1) connecting genes on rows to regulators on columns.
In addition, this produces an unfiltered score matrix (0, ) connecting genes on rows to regulators on columns.

### Requirements

In addition to
python dependencies, this package also requires 
[STAR](https://github.com/alexdobin/STAR),
[sra-tools](http://ncbi.github.io/sra-tools/), 
[bedtools](https://bedtools.readthedocs.io/en/latest/),
[samtools](http://www.htslib.org/), and
[fimo](http://meme-suite.org/doc/fimo.html).

