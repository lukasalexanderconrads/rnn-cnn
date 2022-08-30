## Installation
First, install Python 3.9 or later and pytorch 1.12 or later with cuda.
To install this repository and all dependencies, run
```
pip install -e .
```

## Structure
This repository is structured as follows:
- `experiments` contains folders of .yaml configuration files grouped by data set
  - `blob` refers to the *gaussian-2* data set
  - `blob3c` refers to the *gaussian-3* data set
  - `covtype` refers to the *Forest Covertype* data set
- `scripts` contains the script to train a model
- `src/lab/data` contains data sets and data loaders
- `src/lab/models` contains all our model classes
  - `simple` contains our MLP class for tabular data
  - `recurrent` contains our RNN class for tabular data
- `src/lab/postprocessing` contains tools for evaluation and visualization
- `src/lab/trainer` contains the training script for our experiments

## Experiments
In order to train a model whose configuration is set in `path/to/config.yaml`, run
```
python scripts/train_model.py -c path/to/config.yaml
```
e.g. 
```
python scripts/train_model.py -c experiments/blob/mlp.yaml
```
to train an MLP on the *gaussian-2* data set.

The configuration file options are explained in the `blob` configuration files.