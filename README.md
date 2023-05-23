# Phase-Ordering
It consists of Python3 codes used for the numerical simulation of the our work titled 
"Machine learning based prediction of phase ordering dynamics"


This file "tdgl-chc_ESN.py" can run the proposed ESN architecture for two systems, TDGL and CHC equations.
Before comipling the file one needs to define the model (TDGL or CHC) in line number 76 (or around) with the definition of parameter/hyper-parameters  below that.

The file will create a movie consisting the predicted evolution of the system as well as the target one.

The script also consists commented part to print the predicted data in external file. Those can be uncommented according to the requirements.
