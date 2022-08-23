import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from skimage import data

img = data.coffee()
def lbp_basic(img):
    basic_array = np.zeros(img.shape,np.uint8)
    for i in range(basic_array.shape[0]-1):
        for j in range(basic_array.shape[1]-1):
            basic_array[i,j] = bin_to_decimal(cal_basic_lbp(img,i,j))
    return basic_array

def cal_basic_lbp(img,i,j):#Points larger than the center pixel are assigned a value of 1, and those smaller than the center pixel are assigned a value of 0. The binary sequence is returned
    sum = []
    if img[i - 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j + 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    return sum

def bin_to_decimal(bin):#Binary to decimal
    res = 0
    bit_num = 0 #Shift left
    for i in bin[::-1]:
        res += i << bit_num   # Shifting n bits to the left is equal to multiplying by 2 to the nth power
        bit_num += 1
    return res


def distance3(hog1, hog2):
    sum1 = 0.0
    for i in range(len(hog1)):
        sum1 = sum1 + abs(hog1[i]-hog2[i])

    return sum1

def show_basic_hist(a): #Draw histogram of original lbp
    hist = cv.calcHist([a],[0],None,[256],[0,256])
    hist = cv.normalize(hist,hist)
    print(hist)
    plt.figure(figsize = (8,4))
    plt.plot(hist, color='r')
    plt.xlim([0,256])
    plt.show()

img1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
basic_array = lbp_basic(img1)
show_basic_hist(basic_array)
# plt.figure(figsize=(11,11))
# plt.subplot(1,2,1)
# plt.imshow(img1)
# plt.subplot(1,2,2)
# plt.imshow(basic_array,cmap='Greys_r')
# plt.show()