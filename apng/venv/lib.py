import os,sys
import json
import cv2
# import pydicom as pdc

import rawpy
import imageio

# from extract_image import extract
from PIL import Image
from pathlib import Path

from skimage import data
from skimage import filters
from skimage import color



import math

from PIL import Image

# from resizeimage import resizeimage





def resize_image(img_path, max_size, output_path):
    img = cv2.imread(img_path)
    h, w, c = img.shape

    if c == 4:
        im = Image.open(img_path)
        im = im.convert('RGB')
    else:
        im = Image.open(img_path)




    if(w > max_size):
        resize_height = (max_size / w) * h
        im.thumbnail((max_size, resize_height), Image.ANTIALIAS)
        im.save(output_path, "JPEG")
        # print("---- " + output_path, "JPEG w" + str(w) + " h" + str(h))
        # print("resize w "+output_path, "JPEG")
        return 1

    if (h > max_size):
        resize_width = (max_size / h) * w
        im.thumbnail((resize_width, max_size), Image.ANTIALIAS)
        im.save(output_path, "JPEG")
        # print("---- " + output_path, "JPEG w" + str(w) + " h" + str(h))
        # print("resize h " + output_path, "JPEG")
        return 2

    # print("coppy " + output_path, "JPEG")
    im.save(output_path, "JPEG")

def patch_stride2(img_name, img_path, window_size, output_folder):

    img = cv2.imread(img_path)
    img_w, img_h, c = img.shape


    w_num = math.floor(img_w / (window_size*1.0))+1
    h_num = math.floor(img_h / (window_size*1.0))+1

    size = w_num*window_size, h_num*window_size

    img = Image.open(img_path)
    img = img.resize((h_num*window_size, w_num*window_size), Image.ANTIALIAS)
    img.save('tmpx.png')

    img = Image.open('tmpx.png')

    for i in range(int(w_num/2)):
        for j in range(int(h_num/2)):

            if(i+1<(int(w_num/2))+1 and j+1<(int(h_num/2))+1):
                img_crop = img.crop((j * 2 * window_size,i * 2 * window_size, ((j*2)+1) * window_size,  ((i*2)+1) * window_size))

                ( os.path.join(output_folder, str(i + 1)+ '_'+ str(j + 1)+ '_'+ img_name) )

def patch(img_name, img_path, window_size, output_folder):

    img = cv2.imread(img_path)
    img_w, img_h, c = img.shape


    w_num = math.floor(img_w / (window_size*1.0))+1
    h_num = math.floor(img_h / (window_size*1.0))+1

    size = w_num*window_size, h_num*window_size

    img = Image.open(img_path)
    img = img.resize((h_num*window_size, w_num*window_size), Image.ANTIALIAS)
    img.save('tmpx.png')

    img = Image.open('tmpx.png')
    # print(w_num, h_num)
    for i in range(w_num):
        for j in range(h_num):

            if(i+1<w_num+1 and j+1<h_num+1):
                # print(i * window_size, (i+1) * window_size, j * window_size, (j+1) * window_size)
                img_crop = img.crop((j * window_size,i * window_size, (j+1) * window_size,  (i+1) * window_size))
                # print(img_crop)
                img_crop.save( os.path.join(output_folder, str(i + 1)+ '_'+ str(j + 1)+ '_'+ img_name) )





def segmentation_otsu(img):
    img = color.rgb2gray(img)
    val = filters.threshold_otsu(img)
    mask = img < val
    return mask

def read_raw(file):
    path = file
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()


def get_sub_dir(dir_path):
    PathDicom = dir_path
    lstFilesDCM = []  # create an empty list
    for dirName in os.listdir(PathDicom):
        lstFilesDCM.append(os.path.join(dirName))
    return lstFilesDCM

def get_dmc_file(dir_path):
    PathDicom = dir_path
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(filename))
    return lstFilesDCM

def get_file(dir_path):
    PathDicom = dir_path
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(filename))
    return lstFilesDCM



def get_local_config():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return json.load(open(os.path.join(dirname, "config.json")))


def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2