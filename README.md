# somnotate-vlkuzun

Repo for Harris lab specific requirements utilising somnotate automated vigilance state annotation of EEG/EMG datasets.

Based upon https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011793

## Setup

Ensure conda-forge is natively the channel for install conda environments by setting up environment with 

```conda create -n myenv python=3.10
conda activate myenv
conda config --add channels conda-forge
conda config --set channel_priority strict```

## Installation of required packages

Move into working directory of repo and install required packages via 
```conda activate myenv
conda env update --file environment.yml```

This will ensure all packages are first installed if available on conda environment otherwise installed via pip.