# PHSX815 Spring 2021 Week 5

## Rejection Sampling

This repository contains several types of programs:

- `CalculatePi.x` [C++] 
- `GaussRandom.x` [C++]

and corresponding python versions in `python` directory

### Requirements

In order to compile (by typing `make`) and run the C++ examples, you
need the ROOT package installed (for visualization):
- [ROOT](https://root.cern/) (C++)

The python scripts require the following libraries: `numpy` and `matplotlib`

### Usage

All of the executables can be called from the
command line with the `-h` or `--help` flag, which will print the options

The python scripts can be run with the following options
- `python python/CalculatePi.py -Nsample [# samples]`
- `python python/GaussRandom.py -Nsample [# samples] -range [Xmax] --log --expo`
