### Version 0.3.8

* Fixed bug when saving motif locations with `network_from_motifs_fasta`

### Version 0.3.7

* Improved memory usage for `network_from_motifs_fasta` when a large number of sequences or TFs are used

### Version 0.3.6

* Fix bug in saved location output

### Version 0.3.5

* Fix bug when no regulators are found and saving locations is set.

### Version 0.3.4

* Produce a filtered TF binding table when `--save_filtered_location_data` is set.

### Version 0.3.3

* Correctly produced a TF binding table when `--save_location_data` is set.

### Version 0.3.2

* Corrected a parsing error when reading CisBP PWM files 

### Version 0.3.1

* Added `link_atac_bed_to_genes` module to link specific peaks from a BED file to nearby genes 

### Version 0.3.0

* Finalized model for prior matrix thresholding

### Version 0.2.3

* Added additional messaging and a `--debug` mode

### Version 0.2.2

* Added additional messaging
* Restructured internal memory usage to be more efficient
* Added constraint lists for TFs and genes

### Version 0.2.1

* Parallelized the final information matrix clustering and cutting
* Fixed a bug that included extra inappropriate edges in the final matrix in rare cases

### Version 0.2.0

* Renamed to inferelator-prior
* Fully restructued and parallelized
* Implemented unit testing
* Changed edge selection logic to center around DBSCAN univariate clustering