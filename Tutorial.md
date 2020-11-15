This tutorial is designed to walk through a basic example of motif-based network inference in Yeast.

Set Up Inferelator-Prior
------------------------

Install [anaconda](https://www.anaconda.com/products/individual). 
Create a new environment `conda create --name inferelator_prior` and activate it `conda activate inferelator_prior`.
Install the necessary binaries with `conda install -c bioconda bedtools meme homer star samtools pysam sra-tools`.

#### Install from GitHub

Clone the codebase: 

```git clone https://github.com/flatironinstitute/inferelator-prior.git```

Enter its top-level directory:

```cd inferelator-prior```

Install the inferelator-prior package and any python dependencies

```python setup.py develop --user```

#### Install from PyPi using pip

```python -m pip install inferelator-prior --user```

Build Network
-------------

Change to the example data directory in the github repository ``cd data``, 
or download example data from the [github inferelator-prior repo](https://github.com/flatironinstitute/inferelator-prior/tree/release/data).

First, create a summary of the motif MEME file obtained from [CIS-BP](http://cisbp.ccbr.utoronto.ca/bulk.php):
```
python -m inferelator_prior.motif_information --motif Scer.cisbp.meme --out Scer.cisbp.info.tsv
```

This will summarize the motifs as follows:
```
Motif_ID	Motif_Name	Information_Content	Shannon_Entropy	Length	Consensus
M00001_2.00	ABF1	        15.210	                1.874	        8       TTATCACT
M00002_2.00	AFT2	        7.829	                13.994	        10      NGGGTGTNNN
M00003_2.00	MBP1	        10.704	                6.334	        8       GACGCGTA
M00004_2.00	SWI4	        11.038	                6.441	        8       GACGCGAA
M00005_2.00	XBP1	        11.143	                5.838	        8       TCTCGAAG
M00006_2.00	PHD1	        12.830	                3.840	        8       GMTGCAGG
```

#### Build network from DNA & Motifs Only

```
python -m inferelator_prior.network_from_motifs --motif Scer.cisbp.meme  --out ./Scer_cisbp \
                                                --gtf Saccharomyces_cerevisiae.R64-1-1.UTR.gtf \
                                                --fasta Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa \
                                                --species yeast -c 10
```
Three analyses occur in order:

1) Scan the genome for motifs that match `Scer.cisbp.meme` with the [FIMO](http://meme-suite.org/doc/fimo.html) tool.
2) Score every TF -> Gene edge based on a modified information content
3) Select edges to retain for each TF by clustering TF -> Gene scores and retaining the highest scoring cluster

This writes two files using the prefix provided to `--out`:

`Scer_cisbp_edge_matrix.tsv.gz` is a Genes x TFs boolean (0 & 1) matrix which has 9,973 non-zero edges.
This file has been filtered to only the highest scoring edges

`Scer_cisbp_unfiltered_matrix.tsv.gz` is a Genes x TFs float matrix which has 181,906 non-zero edges.
The values in this file are the scores calculated for each TF -> Gene pair before filtering to the highest scoring edges.

#### Build network from DNA, Motifs & Chromatin Accessibility Data

Constrain the network using the control chromatin accessibility data from [GSM1621339](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1621339).
Note that it is critical that the BED file chromosome names match the GTF and FASTA file chromosome names:
```
python -m inferelator_prior.network_from_motifs --motif Scer.cisbp.meme  --out ./Scer_cisbp \
                                                --gtf Saccharomyces_cerevisiae.R64-1-1.UTR.gtf \
                                                --fasta Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa \
                                                --bed Saccharomyces_cerevisiae.ATAC.GSM1621339.bed \
                                                --species yeast -c 10
```

Here the analysis is repeated, but only chromatin that is accessible (has a peak in the BED file) is considered for mapping.
With this constraint, 7,830 edges are retained in the boolean matrix `Scer_cisbp_edge_matrix.tsv.gz` 
and 61,924 non-zero edges are present in `Scer_cisbp_unfiltered_matrix.tsv.gz`.