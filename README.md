# ml4ils
## Description
A collection of Machine Learning benchmarks for analyzing Incomplete Lineage Sorting in genomic datasets.

simulation parameters and summary statistics: https://docs.google.com/spreadsheets/d/1_egm-zYs2AsZAymv5Ve_Vw_2hYQhaBRa2E3XmDhwqns/edit#gid=0

## Directory Structure

### Code
All simulation scripts are in code/simulate.  ML training and plotting are in code/pylib. Pretrained models and model configurations are in models/.

Miscellaneous scripts for running the pipeline are in code/shell_scripts/ and pbs/ (NOTE: these may contain hard-coded paths.)

### Training Data
All simulations (parameters, gene trees, inferred trees) are stored in a PostGreSQL database.  The summary statististics for the subsets of the database used for training/testing are found in data/simulated/.

### Results
Dataset configuration, model parameters, raw predictions, and performance graphs are in the results/ directory.

## Dependencies and Instructions for Running Code

Python 3.6+.  Required python packages are listed in the code/conda-env.yml file.

### User Guide
The wiki page of this repository (https://github.com/bkrosenz/ml4ils/wiki) contains a complete soup-to-nuts walkthrough example of the inference pipeline.

## TODO List
see https://docs.google.com/spreadsheets/d/1ltwQcEvl_9chGwFglrYC9aERxm0iX5q_MHn_L4LAICY/edit#gid=1386834576
