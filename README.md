# Robust Detection and Modeling of the Major Temporal Arcade in Retinal Fundus Images

Python implementation of a robust MTA detection-modeling algorithm, based on preprint paper "Robust Detection and Modeling of the Major Temporal Arcade in Retinal Fundus Images" (submitted to the journal "MDPI Mathematics" on March 10th, 2022)

<!--[](figures/graphical_abstract.png)-->

## Abstract

The Major Temporal Arcade (MTA) is a critical component of the retinal structure that facilitates clinical diagnosis and monitoring of various ocular pathologies.
Although recent works have addressed the quantitative analysis of the MTA through parametric modeling, their efforts are strongly based on an assumption of symmetry in the MTA shape.
This work presents a robust method for the detection and piecewise parametric modeling of the MTA in fundus images. 
The model consists of a piecewise parametric curve with the ability to consider both symmetric and asymmetric scenarios. In an initial stage, multiple models are built from random blood vessel points taken from the blood-vessel segmented retinal image, following a weighted-RANSAC strategy. 
To choose the final model, the algorithm extracts blood-vessel-width and grayscale-intensity features and merges them to obtain a coarse MTA probability function, which is used to weight the percentage of inlier points for each model. This procedure promotes selecting a model based on points with high MTA probability. 
Experimental results in the public benchmark dataset Digital Retinal Images for Vessel Extraction (DRIVE), for which manual MTA delineations have been prepared, indicate that the proposed method outperforms existing approaches with a balanced Accuracy of 0.7067, Mean Distance to Closest Point of 7.40 pixels, and Hausdorff Distance of 27.96 pixels, while demonstrating competitive results in terms of execution time (9.93 seconds per image).

## Prerequisities

This code has been tested with python 3.6.10

The following dependencies are needed:

- numpy >= 1.19.1
- Pillow >=7.2.0
- opencv >= 4.4.0
- scikit-learn >= 0.23.1

See requirements.txt for more details.

Additionally, the [DRIVE](https://drive.grand-challenge.org/) dataset is needed.  
We are not allowed to provide the dataset here, but it can be freely downloaded from the official website by joining the challenge. 
The original images should be included in the path "./images/images".
The Field-of-View mask images should be included in the path "./images/fov".

The vessel-probability images obtained from a vanilla U-Net model are included in the path "./images/unet".

The ground-truth MTA delineations for the DRIVE dataset are freely available in this 
[link](http://personal.cimat.mx:8181/~ivan.cruz/Journals/MTA_drive_files/MTA_images.zip).
These images should be included in the path "images/mta_annotations"  

The "./images" folder should have the following structure:

```
images
│
└───fov
└───images
└───mta_annotations
└───unet
```

