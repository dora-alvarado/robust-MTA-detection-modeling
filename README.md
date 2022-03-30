# Robust Detection and Modeling of the Major Temporal Arcade in Retinal Fundus Images

Python implementation of a robust MTA detection-modeling algorithm, based on preprint paper "Robust Detection and Modeling of the Major Temporal Arcade in Retinal Fundus Images" (submitted to the journal "MDPI Mathematics" on March 10th, 2022)

<!--[](figures/graphical_abstract.png)-->

## Abstract

The Major Temporal Arcade (MTA), the thickest vessel in the retina, has been used as an indicator of the severity for 
many ocular pathologies for physicians. In this work, a novel method for the automatic detection and numerical modeling 
of the MTA is presented. The method uses a quadratic-spline approximation with a weighted Random Sample Consensus (RANSAC) 
scheme for estimating a MTA Probability Map, based on both a Distance-Transform-based vessel thickness image and a 
foreground-location map. The performance of the proposed method is evaluated in terms of vessel detection, vessel skeleton 
detection, and closeness between the numerical modeling and the ground-truth delineation of the MTA, using the public 
benchmark dataset Digital Retinal Images for Vessel Extraction (DRIVE) and manual MTA delineations provided by an expert 
ophthalmologist. The computational results indicate that the automatic detection and numerical modeling of the MTA achieved  
by the proposed method outperforms existing approaches with a balanced Accuracy of 0.7067, Mean Distance to Closest Point 
of 7.40 pixels, and Hausdorff Distance of 27.96 pixels, while demonstrating competitive results in terms of execution time 
(9.93 seconds per image).

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

