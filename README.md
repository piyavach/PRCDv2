# PRCDv2
Arrangement of the Object Image Extraction watching youtube sample https://www.youtube.com/watch?v=KG9czA0MueM&t=24s

We implement our technique to Arrangement of the Object Image Extraction. 
It is called Patch Relational Covariance Descriptor version 2 (PRCD2).
This work is based on https://link.springer.com/chapter/10.1007/11744047_45.
If you are interested and use it in your work, please cite and follow this work. https://dl.acm.org/doi/abs/10.1145/3411174.3411200.
In case you are using please cite at PRCDv1 https://dl.acm.org/doi/abs/10.1145/3411174.3411200.

Note: This work is implemented for my education and study without money profit.



PRCDv2 is a technique to evaluate similarity between two object  arrangements.
This technique is inspired based on patching or windows sliding techniques such a HOG and combined with the region covariance technique.
We get a good nDCG score at 0.97 and this works fine when the setting up environment is like a good providing studio such as COIL-100.
The main citations is in the below README.
*If you want to cite our work, you could please cite our whole main citations, currently.
*We note that in future. If our work PRCDv2 are published, we update a primary citation to our mainly citations section below.

Getting started

Note: we run programme over the python 3.6 and intellij ide

For PRCD code
1. include the libarary as following
import json

import shutil

import statistics

import time

import reg_cov

from PIL import Image

from lib import patch

from lib import get_sub_dir

from lib import get_file

import cv2

import numpy as np

import math

import sys

import os

import numpy

2. fixed the path of the tmp file
such as 
#parameter and data temporary path example. Those paths we set on our computer for temporary image such gray scale, patching image and etc. for illustration and clarify samples.

_proj_path = "C:\\apng\\" #### <---change this for your project path and run ***

We fixed this as our computer paths. This necessary to be change in the code as you computer path.



3 the running file is on the c:\apng\prcd.py <--- this project is in our computer path.

note: 
function to create PRCD descriptor

json_descriptor("car1", img, descriptor_relation)

function to return PRCD illustration image is at 

prcd_illustrate(src+src_no+".jpg", descriptor_relation["car1"]["z"], _max_similar_t) #path keep PRCD illustration path (_illus = "C:\\apng\\venv\\_illus\\")

function to determine the similarity between two images

distance2(descriptor_relation["car1"]




reference

M. Jamshed, S. Parvin, and S. Akter, “Significant HOG-Histogram of Oriented Gradient Feature Selection for Human Detection,” International Journal of Computer Applications, vol. 132, no. 17, pp. 20–24, Dec. 2015, doi:10.5120/IJCA2015907704. 

O. Tuzel, F. Porikli, and P. Meer, “Region Covariance: A Fast Descriptor for Detection and Classification,” in Proceedings of the 9th European conference on Computer Vision - Volume Part II, Springer-Verlag, 2006, pp. 589–600. doi: 10.1007/11744047_45. 

P. Khunsongkiet, “Patch Relational Covariance Distance Similarity Approach for Image Ranking in Content-Based Image Retrieval,” 2020. http://www.iccfi.org/Files/2019/July Bangkok Conference Abstract.pdf 

S. A. Nene, S. K. Nayar, and H. Murase, “Columbia Object Image Library (COIL-100).” 

Katerenchuk, D. & Rosenberg, A. Rankdcg rank–ordering evaluation measure (2016). Information Retrieval, Social and Information Networks, LREC, 2016

https://cocodataset.org/
