# DoE Package

## RESPONSIBILITY:
This package is responsible for managing the information required to generate the DoE data table and used by the simulator, the machine learning and the optimization packages.

## Requirements:
* shall automatically generate data points using several sampling methods
* shall allow users to input design parametrs as strings, constants, or ranges
* shall allow user to choose sampling method and sampling size
* shall apply sampling only to parameters defined as value ranges, and distinguish between ranges of integer or ranges of doubles.
* shall allow users to input certain parameters as nested data structures (loosly defined dictionaries) for the definition of materials as input parameters
* shall generate doe table (a flat representation of all input parameters) on request by the user.