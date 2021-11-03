# Simulator Package

## RESPONSIBILITY:

This package provides a library of functions for common operations required by the pre-processing, running and post-processing steps for simulation packages (e.g. Abaqus, Fenics, etc.)

## REQUIREMENTS:

* Shall generate DoE data table  in formats that can be easily ingested by the simulator package of choice.
* shall provide a strucure to organize operations for every of the following steps: pre-processing, running, and post-processing for every simulation package, but not all steps are compulsory for every simulation packages
* shall request users to provide pre-processing (data translation for simulation package), running (simulation problem and solution) and pre-processing (data translation from simulation package) instructions as scripts for every case and simulation package.
* shall enable developers to exend its funcionality by implementing connectors (translations of DoE table, simulation problems and solutions) for well-known simulation engines (e.g. Abaqus, Fenics)
* shall collect simulation output to extend DoE data table.
