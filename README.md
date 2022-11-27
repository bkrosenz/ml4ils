# ml4ils
## Description
A collection of Machine Learning benchmarks for analyzing Incomplete Lineage Sorting in genomic datasets.

## Directory Structure

### Code
All simulation scripts are in code/simulate.  ML training and plotting are in code/pylib. Pretrained models and model configurations are in models/.

Miscellaneous scripts for running the pipeline are in code/shell_scripts/ and pbs/ (NOTE: these may contain hard-coded paths.)

### Training Data
Training data was simulated with \verb+ms+ and \verb+seq-gen+, and gene trees inferred with \verb+RaXML+ and \verb+IQTREE+.  Simulation parameters and summary statistics are described in the paper.  All simulations (parameters, gene trees, inferred trees) are stored in a PostGreSQL database.  Scripts for generating data, reading from the database, and computing summary statistics are included in this repository.

Example summary statistics files for training/testing are found in \verb+data/simulated/+.  Complete simulated training files are available on data dryad at doi:10.5061/dryad.1ns1rn8xx.


### Results
Dataset configuration, model parameters, raw predictions, and performance graphs are in the results/ directory.

## Dependencies and Instructions for Running Code

Python 3.6+.  Required python packages are listed in the code/conda-env.yml file.

### User Guide
The wiki page of this repository (https://github.com/bkrosenz/ml4ils/wiki) contains a complete soup-to-nuts walkthrough example of the inference pipeline.
