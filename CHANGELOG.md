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