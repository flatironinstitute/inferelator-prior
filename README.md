# inferelator-prior

This is a set of pipelines to create expression and prior matrices for network inference. They are designed to create
data that is compatible with the [inferelator](https://github.com/flatironinstitute/inferelator) package. 

### Usage

    python -m inferelator_prior.network_from_motifs
    usage: network_from_motifs.py -m motif_PWM_file.meme
                                  -f genome_fasta_file.fasta
                                  -g genome_annotation_file.gtf
                                  -o ~/output/path/prefix
                                  --species {yeast,fly,mouse,human}]
                                  
This requires a motif PWM database (`-m PATH`), 
a genome to search (both sequence as a FASTA `-f PATH` and annotations `-g PATH`),
and an output prefix for several files (`-o PATH`).
In addition, default settings for a specific species can be set with (`--species`).

### Requirements

In addition to
python dependencies, this package also requires 
[STAR](https://github.com/alexdobin/STAR),
[sra-tools](http://ncbi.github.io/sra-tools/), 
[bedtools](https://bedtools.readthedocs.io/en/latest/),
[samtools](http://www.htslib.org/), and
[fimo](http://meme-suite.org/doc/fimo.html).

