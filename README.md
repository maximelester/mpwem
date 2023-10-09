# MPWEM
Moiré plane wave expansion model
Python library and example file.
(c) Maxime Le Ster, 2023

## Introduction

Full paper describing the MPWEM can be found (DOI: to be disclosed)

The python library (MPWEM.py) allows to simulate a scanning tunneling microscopy (STM) image for arbitrary layer geometries and twist angles using the MPWEM.
The main ingredients of the MPWEM are:

1) Plane wave parameters of the substrate layer
2) Plane wave parameters of the top layer
3) MPWEM parameters (mu, tau, a0, eta)

## Files 

The example.py file shows in detail how to use the MPWEM.py functions leading to the STM simulated image (includes plotting routines).
The file should be adequately commented for detailed understanding at each step of the simulation procedure.
The physical system in the example file is alpha-bismuthene (aBi) on molybdenum disulfide (MoS2) used in the seminal paper.
The raw STM data is contained in the bimos2_rawdata.txt.

![figure](https://github.com/maximelester/mpwem/assets/75083058/dea2e55a-db10-4cfe-968b-d9d0ae0f5bf5)

## Dependencies

The following python libraries are required: numpy, matplotlib, scipy and time. The extra python file (symbols.py) is a library of special characters and only used for display purposes.

## Contact

For more information, please contact me via email at maxime.lester@fis.uni.lodz.pl (or pawel.kowalczyk@uni.lodz.pl)
If this model is useful, please cite the reference paper (DOI: to be disclosed).
