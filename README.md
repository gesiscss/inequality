# Individual and gender inequality in computer science: A career study of cohorts from 1970 to 2000

This repository houses the code to fully reproduce the study by Lietz et al. (forthcoming). Read the preprint [here](https://arxiv.org/abs/2311.04559).

## Computational environment

The code is developed for the Python and package versions specified in the [environment](environment.yml) file. To guarantee reproducibility, you can set up a local environment by following these steps:

1. Install the [Anaconda Distribution](https://www.anaconda.com/download)
2. Download the [environment](environment.yml) file into your user directory
3. In your user directory, run this command to set up a local environment: `conda env create -f environment.yml`
4. Activate this environment to work with it: `conda activate inequality`

You can also run this code in the cloud by clicking on this button:

[![Binder](https://mybinder.org/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/gesiscss/inequality/HEAD)

A docker image is [here](https://hub.docker.com/r/gesiscss/binder-r2d-g5b5b759-gesiscss-2dinequality-d3a6f4).

## Execution

To reproduce the study, first execute the [preprocessing](code/1_preprocessing.ipynb) notebook. This will write files. Once these files are written, you can execute the other notebooks to produce all figures and tables in the paper.

## References

This is the study that is getting reproduced:

```
@article{lietz_individual_forthcoming,
  author={Haiko Lietz and Mohsen Jadidi and Daniel Kostic and Milena Tsvetkova and Claudia Wagner},
  title={Individual and gender inequality in computer science: {A} career study of cohorts from 1970 to 2000},
  journal={Quantitative Science Studies},
  year={forthcoming}
}
```

This is the data used in the study:

```
@misc{lietz_computer_2023,
author = {Haiko Lietz and Mohsen Jadidi and Daniel Kostic and Claudia Wagner},
title = {Computer Science (1970-2014)},
year = {2023},
howpublished = {GESIS, K{\"o}ln. Data file version 1.0.0, https://doi.org/10.7802/2642},
doi = {10.7802/2642}
}
```

Please cite this code repository as follows:

```
@misc{kostic_inequality_2023,
  author = {Daniel Kostic and Haiko Lietz and Mohsen Jadidi and Claudia Wagner},
  title = {Individual and gender inequality in computer science: {A} career study of cohorts from 1970 to 2000},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gesiscss/inequality/}}
}
```

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)
