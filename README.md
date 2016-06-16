# shelter-animals-kaggle

[Shelter Animal Outcomes Kaggle competiton](https://www.kaggle.com/c/shelter-animal-outcomes)

> Every year, approximately 7.6 million companion animals end up in US shelters. Many animals are given up as unwanted by their owners, while others are picked up after getting lost or taken out of cruelty situations. Many of these animals find forever families to take them home, but just as many are not so lucky. 2.7 million dogs and cats are euthanized in the US every year. **Using a dataset of intake information including breed, color, sex, and age from the Austin Animal Center, we're asking Kagglers to predict the outcome for each animal.**

## Overview

## Scratchwork

* [`1-exploring.ipynb`](https://github.com/jakesherman/shelter-animals-kaggle/blob/master/1-exploring.ipynb) - Exploratory analysis  
* [`2-dog-breeds.ipynb`](https://github.com/jakesherman/shelter-animals-kaggle/blob/master/2-dog-breeds.ipynb) - Classifying dog breeds into clusters from the [American Kennel Club](http://www.akc.org/dog-breeds/)  
* [`3-feature-engineering.ipynb`](https://github.com/jakesherman/shelter-animals-kaggle/blob/master/3-feature-engineering.ipynb) - Feature engineering  
* [`4-modeling.ipynb`](https://github.com/jakesherman/shelter-animals-kaggle/blob/master/4-modeling.ipynb) - Model building and selection, making the submission

## Usage

First, make sure that [Miniconda](http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install) is installed on your machine. Next, run the following in a directory of your choosing to clone this repo and create a conda environment w/ Python 2.7 and all of the necessary packages installed

```bash
git clone https://github.com/jakesherman/shelter-animals-kaggle.git
cd shelter-animals-kaggle
conda env create
source activate shelter-animals
```

Re-create my best submission with

```bash
python project.py
```
