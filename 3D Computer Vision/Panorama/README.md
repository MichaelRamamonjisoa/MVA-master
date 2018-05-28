# Panorama

This assignment illustrates the 8 points methods for homography correction between two views of the same scene. The world is assumed to be flat, and the two views are related by a rotation transformation.

## Setup Panorama:

* Make sure [*Imagine++*](http://imagine.enpc.fr/~monasse/Imagine++/) is installed.

* Build with CMake, using the following command lines 
- $ mkdir _build && cd _build 
- $ cmake ..
- $ make

## Run Panorama
Use the following command line to run Panorama (in the _build folder):
* $ ./Panorama

## Instructions

### Step 1:
The user should first locate similar points in both images by clicking alternatively on the two images.
At least four pairs (which is a total of 8 clicks) are needed to reconstruct the panorama correctly.

The user can stop this step by right-clicking at the final pair.

### Step 2:

The homography matrix is automatically computed is return in the terminal.

### Step 3: 

The reconstructed panorama in displayed.

