# ml4ils
## Description
A collection of Machine Learning benchmarks for analyzing Incomplete Lineage Sorting in genomic datasets.

simulation parameters and summary statistics: https://docs.google.com/spreadsheets/d/1_egm-zYs2AsZAymv5Ve_Vw_2hYQhaBRa2E3XmDhwqns/edit#gid=0

## Directory Structure

### Training Data
All simulations (parameters, gene trees, inferred trees) are stored in a PostGreSQL database.  The summary statististics for the subsets of the database used for training/testing are found in data/simulated/.

### Results
Dataset configuration, model parameters, raw predictions, and performance graphs are in the results/ directory.

## Dependencies and Instructions for Running Code

Python 3.6+.  Required python packages are listed in the code/conda-env.yml file.

## TODO List
see https://docs.google.com/spreadsheets/d/1ltwQcEvl_9chGwFglrYC9aERxm0iX5q_MHn_L4LAICY/edit#gid=1386834576
