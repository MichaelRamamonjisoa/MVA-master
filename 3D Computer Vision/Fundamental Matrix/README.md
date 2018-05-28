# Fundamental

This assignment is an introduction to Fundamental matrix computation, using [*RANSAC*](https://en.wikipedia.org/wiki/Random_sample_consensus) algorithm. It also makes use of the [*VLFeat*](http://www.vlfeat.org/) toolbox for [*SIFT*] features detection. The goal of this assignment is to compute the Fundamental matrix from SIFT matches between two images using the RANSAC algorithm, and then display epipolar lines in each view.

## Setup Fundamental
* Make sure [*Imagine++*](http://imagine.enpc.fr/~monasse/Imagine++/) is installed.

* Build with CMake, using the following command lines 
- $ mkdir _build && cd _build 
- $ cmake ..
- $ make

## Run Fundamental
Use the following command line to run Panorama (in the _build folder):
* $ ./Fundamental

## Instructions

### Step 1:
The first step is automatic. It detects the SIFT keypoints and matches between the two images.

The user should then click on the image to move to the next step.

### Step 2:

The fundamental matrix is automatically computed as follows:
a) RANSAC algorithm to compute a first fundamental matrix F based on 8 best matches
b) Least square minimization refinement of F based on the inliers found in a)

The best matches inliers are displayed in both images.

The user should then click on the image to move to the next step.

### Step 3: 

The user should click on a point of any image to display the associated epipolar line in the other image.
Repeat as wanted.


### Exit: 
To exit the program, right click in the image, then click again to exit.
