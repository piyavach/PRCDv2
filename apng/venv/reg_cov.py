import os,glob
import numpy as np
import sys

from scipy import linalg
import scipy.sparse as sparse
import PIL
from PIL import Image
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2
import math
import json
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import operator
from scipy import stats

from os.path import dirname, abspath
from sklearn.metrics.pairwise import euclidean_distances

import time


np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def experiment(test_image, json_cbir, patch_train_input_folder,
               patch_train_output_reg_folder,
               sorted_human,
               image_list):




    # closest patch
    f_indexs = {}
    for f_index in json_cbir.keys():
        for a in json_cbir[f_index].copy():
            f_indexs[a] = f_index

    result_to_html = {}

    # normal
    print("--- reg cov old")
    min_pair_o, min_value_o, result_list = closest_regcov_mst(test_image,
                                                              json_cbir,
                                                              os.path.join(patch_train_input_folder),
                                                              f_indexs)
    sorted_by_value = sorted(result_list.items(), key=lambda kv: kv[1])
    print(json.dumps(sorted_by_value))
    result_to_html["regcov"] = sorted_by_value

    ord = []
    for i in range(len(sorted_by_value)):
        ord.append(sorted_by_value[i][0])
    print(ap(sorted_human, ord))

    # patch
    print("--- reg cov average patch")
    min_pair, min_value, result_list = closest_mst(test_image, json_cbir, patch_train_output_reg_folder,
                                                           f_indexs,
                                                           t=50)
    sorted_by_value = sorted(result_list.items(), key=lambda kv: kv[1])
    print(json.dumps(sorted_by_value))
    result_to_html["regcov_patch_avg"] = sorted_by_value

    ord = []
    for i in range(len(sorted_by_value)):
        ord.append(sorted_by_value[i][0])
    print(ap(sorted_human, ord))

    # patch2
    print("--- reg cov sum patch")
    min_pair, min_value, result_list = closest_mst2(test_image, json_cbir,
                                                            patch_train_output_reg_folder, f_indexs, t=50)
    sorted_by_value = sorted(result_list.items(), key=lambda kv: kv[1])
    print(json.dumps(sorted_by_value))
    result_to_html["regcov_patch_sum"] = sorted_by_value

    ord = []
    for i in range(len(sorted_by_value)):
        ord.append(sorted_by_value[i][0])
    print(ap(sorted_human, ord))

    # relation
    reg_D_relation = create_D_patch_matrix(image_list, patch_train_output_reg_folder, t=sys.float_info.max)

    result_list = closest_mst_relation_matrix(test_image, reg_D_relation, t=sys.float_info.max)
    sorted_by_value = sorted(result_list.items(), key=lambda kv: kv[1])
    print("--- relation")
    print(json.dumps(sorted_by_value))
    result_to_html["regcov_patch_relation"] = sorted_by_value

    ord = []
    for i in range(len(sorted_by_value)):
        ord.append(sorted_by_value[i][0])


    print(ap(sorted_human, ord))

    return result_to_html




def ap(actual, predicted):
    print(actual)
    print(predicted)
    hit = []

    for i in range(len(actual)):

        if actual[i] == predicted[i]:
            hit.append(1)
        else:
            hit.append(0)

    # print(hit)

    precision = []
    recall = []

    for i in range(len(hit)):

        #precision

        tp = 0.0
        n1 = i+1

        for j in range(i+1):
            if hit[j] == 1:
                tp = tp + 1

        p = tp/n1
        precision.append(p)

        #recall
        # n2 = 0.0
        #
        # for j in range(len(hit)):
        #     if hit[j] == 1:
        #         n2 = n2 + 1
        #
        # r = tp/n2
        # recall.append(r)




    # delta_recall = []
    # recall_before = 0.0
    # for i in range(len(recall)):
    #     delta_recall.append(recall[i]-recall_before)
    #     recall_before = recall[i]



    result = 0.0
    for i in range(len(precision)):
        # print(str(precision[i]) + " " + str(delta_recall[i]))
        result = result + (precision[i])

    return result/len(precision)




def ap_with_recall(actual, predicted):
    # print(actual)
    # print(predicted)
    hit = []

    for i in range(len(actual)):

        if actual[i] == predicted[i]:
            hit.append(1)
        else:
            hit.append(0)

    # print(hit)

    precision = []
    recall = []

    for i in range(len(hit)):

        #precision

        tp = 0.0
        n1 = i+1

        for j in range(i+1):
            if hit[j] == 1:
                tp = tp + 1

        p = tp/n1
        precision.append(p)

        #recall
        n2 = 0.0

        for j in range(len(hit)):
            if hit[j] == 1:
                n2 = n2 + 1

        r = tp/n2
        recall.append(r)




    delta_recall = []
    recall_before = 0.0
    for i in range(len(recall)):
        delta_recall.append(recall[i]-recall_before)
        recall_before = recall[i]



    result = 0.0
    for i in range(len(recall)):
        # print(str(precision[i]) + " " + str(delta_recall[i]))
        result = result + (precision[i]*delta_recall[i])

    return result/len(recall)



    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        print(p)
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)





def normal_regcov01(x):
    return math.exp(x)/(1+math.exp(x))



def show_img(img):
    plt.imshow(img, cmap=plt.cm.bone)
    plt.show(block=True)

def permute(I, dimension_num_array):
    return np.transpose(I[:, :, None], dimension_num_array)

def RelationCovarianceDescriptor(F):
    fc = F.shape
    f = np.double(np.zeros((1, fc)))

    i = 0
    for n in range(fc):
        f[0][i] = F[fc]
        i = i + 1



    C = np.cov(f)
    return C

def FeatureImage(I):
    P = np.transpose(np.array(I), (2, 0, 1))
    d, n, m = P.shape
    F = np.uint8(np.zeros(shape=(3, n, m)))




    for d in range(len(P)):

        for x in range(len(P[d])):
            for y in range(len(P[d][x])):
                F[d][x][y] = P[d][x][y]
    return F


def FeatureImage2(I):
    P = np.transpose(np.array(I), (2, 0, 1))
    d, n, m = P.shape
    F = np.int8(np.zeros(shape=(9, n, m)))

    grad1 = np.int8(cv2.Sobel(np.array(I), cv2.CV_64F, 1, 0))
    grad2 = np.int8(cv2.Sobel(np.array(I), cv2.CV_64F, 0, 1))
    grad1 = np.transpose(np.int8(grad1), (2, 0, 1))
    grad2 = np.transpose(np.int8(grad2), (2, 0, 1))

    for d in range(len(P)):

        for x in range(len(P[d])):
            for y in range(len(P[d][x])):
                F[d][x][y] = P[d][x][y]
                F[d + 3][x][y] = grad1[d][x][y]

                # print(F.shape)
                # print(d)
                # print(x)
                # print(y)
                F[d + 6][x][y] = grad2[d][x][y]
    return F

def FeatureImage3(I):
    I = I.convert('L')
    P = np.transpose(np.array(I))
    n, m = P.shape
    F = np.int8(np.zeros(shape=(3, n, m)))

    grad1 = np.int8(cv2.Sobel(np.array(I), cv2.CV_64F, 1, 0))
    grad2 = np.int8(cv2.Sobel(np.array(I), cv2.CV_64F, 0, 1))
    grad1 = np.transpose(np.int8(grad1))
    grad2 = np.transpose(np.int8(grad2))

    for d in range(1):
        for x in range(len(P)):
            for y in range(len(P[x])):
                F[0][x][y] = P[x][y]
                F[d + 1][x][y] = grad1[x][y]
                F[d + 2][x][y] = grad2[x][y]


    return F

def RegionCovarianceDescriptor(F):
    # print(F)
    d, x, y = F.shape
    f = np.double(np.zeros((d, x * y)))

    # print(F)


    i = 0
    for a in range(x):
        for b in range(y):
            for n in range(d):
                f[n][i] = F[n][a][b]
            i = i + 1


    C = np.cov(f)


    return C


def CovarianceDistance(C1, C2):
    D, vec = linalg.eig(C1, C2)
    D = np.diag(D)

    # print(np.diag(np.absolute(D)))
    try:
        d = sum(np.log(np.diag(np.absolute(D))) ** 2)

        if d == float("inf") or math.isnan(d):

            D, vec = linalg.eig(C2, C1)
            D = np.diag(D)

            d = sum(np.log(np.diag(np.absolute(D))) ** 2)

            if d == float("inf") or math.isnan(d):
                d = sys.float_info.max
        # print(d)
    except RuntimeWarning:
        d = sys.float_info.max

    return d





#-----------------



def closest_regcov_mstx(image_name1, image_name_sorting_list, image_path, f_indexs_map):
    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}

    I1 = Image.open( os.path.join(image_path, f_indexs_map[image_name1], image_name1+".JPEG") )
    C1 = RegionCovarianceDescriptor(FeatureImage2(I1))

    # print(image_name_sorting_list)
    image_name_sorting_list_tmp = []
    for key in image_name_sorting_list.keys():
        for img in image_name_sorting_list[key]:
            image_name_sorting_list_tmp.append(img)


    image_name_sorting_list = image_name_sorting_list_tmp

    for i in range(len(image_name_sorting_list)):
        image_name2 = image_name_sorting_list[i]
        print(os.path.join(image_path,f_indexs_map[image_name2], image_name2+".JPEG"))
        I2 = Image.open( os.path.join(image_path,f_indexs_map[image_name2], image_name2+".JPEG") )
        C2 = RegionCovarianceDescriptor(FeatureImage2(I2))

        similarity_mean = CovarianceDistance(C1, C2)

        result_list[image_name2] = similarity_mean

        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list


def closest_regcov_mst(image_name1, image_name_sorting_list, image_path, f_indexs_map):
    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}

    I1 = Image.open( os.path.join(image_path, f_indexs_map[image_name1], image_name1+".JPEG") )
    C1 = RegionCovarianceDescriptor(FeatureImage2(I1))

    # print(image_name_sorting_list)
    image_name_sorting_list_tmp = []
    for key in image_name_sorting_list.keys():
        for img in image_name_sorting_list[key]:
            image_name_sorting_list_tmp.append(img)


    image_name_sorting_list = image_name_sorting_list_tmp

    for i in range(len(image_name_sorting_list)):
        image_name2 = image_name_sorting_list[i]
        print(os.path.join(image_path,f_indexs_map[image_name2], image_name2+".JPEG"))
        I2 = Image.open( os.path.join(image_path,f_indexs_map[image_name2], image_name2+".JPEG") )
        C2 = RegionCovarianceDescriptor(FeatureImage2(I2))

        similarity_mean = CovarianceDistance(C1, C2)

        result_list[image_name2] = similarity_mean

        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list





#-----------------

#patch cov
def hcompare_patch_similarity(C1_patchs, C2_patchs, t):
    similarity_mean = 0.0
    n = 0.0

    list = []

    if(len(C1_patchs)<len(C2_patchs)):
        Cs1 = C2_patchs
        Cs2 = C1_patchs
    else:
        Cs1 = C1_patchs
        Cs2 = C2_patchs


    for key1 in Cs1.keys():
        min_d = t  # not similar
        for key2 in Cs2.keys():
            d = CovarianceDistance(Cs1[key1], Cs2[key2])
            # d1 = CovarianceDistance(C2_patchs[key2], C1_patchs[key1])
            # print(str(d) + "-" + str(d1))

            if min_d > d:
                min_d = d

        if (d != sys.float_info.max):
            list.append(min_d)

            similarity_mean = similarity_mean + min_d
            n = n + 1.0


    if n != 0.0:
        return similarity_mean/n
    else:
        return t # for high value for the high dissimilar

# patch
def closest_mstx(image_name1, image_name_sorting_list, json_covmatrix_patch_list, f_indexs, t):

    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}
    C1_patchs = json_covmatrix_patch_list[image_name1]

    for image_name2 in image_name_sorting_list:

        print(image_name1 + " --- " + image_name2)

        C2_patchs = json_covmatrix_patch_list[image_name2]
        similarity_mean = hcompare_patch_similarity(C1_patchs, C2_patchs, t)
        result_list[image_name2] = similarity_mean
        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list

def closest_mst(image_name1, image_name_sorting_list, patch_train_output_reg_folder, f_indexs, t):
    json_covmatrix_patch_list1 = {}
    if (f_indexs == []):
        json_covmatrix_patch_list1 = json.load(open(os.path.join(patch_train_output_reg_folder, 'result' + image_name1.split("_")[0] + '.json')))
    else:
        json_covmatrix_patch_list1 = json.load(open(os.path.join(patch_train_output_reg_folder, 'result' + f_indexs[image_name1] + '.json')))

    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}

    C1_patchs = json_covmatrix_patch_list1[image_name1]

    image_name_sorting_list_tmp = []
    if (f_indexs == []):
        image_name_sorting_list_tmp = image_name_sorting_list
    else:
        image_name_sorting_list_tmp = []
        for key in image_name_sorting_list.keys():
            for img in image_name_sorting_list[key]:
                image_name_sorting_list_tmp.append(img)

    image_name_sorting_list = image_name_sorting_list_tmp

    for i in range(len(image_name_sorting_list)):
        image_name2 = image_name_sorting_list[i]

        json_covmatrix_patch_list2 = {}
        if(f_indexs == []):
            head, tail = os.path.split(image_name2)
            tail = tail.split(".")[0]
            image_name2 = tail
            json_covmatrix_patch_list2 = json.load(
                open(os.path.join(patch_train_output_reg_folder, 'result' + tail.split("_")[0] + '.json')))
        else:
            json_covmatrix_patch_list2 = json.load(
                open(os.path.join(patch_train_output_reg_folder, 'result' + f_indexs[image_name2] + '.json')))
        C2_patchs = json_covmatrix_patch_list2[image_name2]
        similarity_mean = hcompare_patch_similarity(C1_patchs, C2_patchs, t)
        result_list[image_name2] = similarity_mean

        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list






#patch cov
def hcompare_patch_similarity2(C1_patchs, C2_patchs, t):
    d_list = []
    similarity_mean = 0.0
    n = 0.0

    list = []

    if(len(C1_patchs)<len(C2_patchs)):
        Cs1 = C2_patchs
        Cs2 = C1_patchs
    else:
        Cs1 = C1_patchs
        Cs2 = C2_patchs


    for key1 in Cs1.keys():
        min_d = t  # not similar
        for key2 in Cs2.keys():
            d = CovarianceDistance(Cs1[key1], Cs2[key2])
            # d1 = CovarianceDistance(C2_patchs[key2], C1_patchs[key1])
            # print(str(d) + "-" + str(d1))

            if min_d > d:
                min_d = d

        if(d != sys.float_info.max):
            list.append(min_d)
            d_list.append(min_d)
            similarity_mean = similarity_mean + min_d
            n = n + 1.0


    if n != 0.0:
        return similarity_mean, d_list
    else:
        return t, d_list # for high value for the high dissimilar

# patch
def closest_mst2x(image_name1, image_name_sorting_list, json_covmatrix_patch_list, f_indexs, t):
    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}

    C1_patchs = json_covmatrix_patch_list[image_name1]

    for image_name2 in image_name_sorting_list:

        print(image_name1 + " --- " + image_name2)

        C2_patchs = json_covmatrix_patch_list[image_name2]
        similarity_mean, d_list = hcompare_patch_similarity2(C1_patchs, C2_patchs, t)
        result_list[image_name2] = similarity_mean

        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list

def closest_mst2(image_name1, image_name_sorting_list, patch_train_output_reg_folder, f_indexs, t):
    json_covmatrix_patch_list1 = {}
    if (f_indexs == []):
        json_covmatrix_patch_list1 = json.load(
            open(os.path.join(patch_train_output_reg_folder, 'result' + image_name1.split("_")[0] + '.json')))
    else:
        json_covmatrix_patch_list1 = json.load(
            open(os.path.join(patch_train_output_reg_folder, 'result' + f_indexs[image_name1] + '.json')))

    min_pair = ""
    min_value = sys.float_info.max

    result_list = {}

    C1_patchs = json_covmatrix_patch_list1[image_name1]

    image_name_sorting_list_tmp = []
    if (f_indexs == []):
        image_name_sorting_list_tmp = image_name_sorting_list
    else:
        for key in image_name_sorting_list.keys():
            for img in image_name_sorting_list[key]:
                image_name_sorting_list_tmp.append(img)

        image_name_sorting_list = image_name_sorting_list_tmp

    for i in range(len(image_name_sorting_list)):
        image_name2 = image_name_sorting_list[i]

        json_covmatrix_patch_list2 = {}
        if (f_indexs == []):
            head, tail = os.path.split(image_name2)
            tail = tail.split(".")[0]
            image_name2 = tail
            json_covmatrix_patch_list2 = json.load(
                open(os.path.join(patch_train_output_reg_folder, 'result' + tail.split("_")[0] + '.json')))
        else:
            json_covmatrix_patch_list2 = json.load(
                open(os.path.join(patch_train_output_reg_folder, 'result' + f_indexs[image_name2] + '.json')))

        C2_patchs = json_covmatrix_patch_list2[image_name2]
        similarity_mean, d_list = hcompare_patch_similarity2(C1_patchs, C2_patchs, t)
        result_list[image_name2] = similarity_mean

        if similarity_mean < min_value:
            min_pair = image_name2
            min_value = similarity_mean

    return min_pair, min_value, result_list








#-----------------

#relation coco



def closest_mst_relation_matrix_coco(image_name1, images, result_D_relation_folder, t):
    result_list = {}

    # m1 = image_name_relation[image_name1]["m"]

    for image_name2 in images:
        d1 = _closest_mst_relation_matrix_coco(image_name1, image_name2, result_D_relation_folder, t)
        d2 = _closest_mst_relation_matrix_coco(image_name2, image_name1, result_D_relation_folder, t)

        if d1 < d2:
            result_list[image_name2] = d1
        else:
            result_list[image_name2] = d2

        # print(image_name1 + " --- " + image_name2, result_list[image_name2])

    return result_list

def _closest_mst_relation_matrix_coco(image_name1, image_name2, result_D_relation_folder, t):
    json_image_patch_relation1 = json.load(open( os.path.join(result_D_relation_folder, image_name1+'.json')))
    json_image_patch_relation2 = json.load(open(os.path.join(result_D_relation_folder, image_name2 + '.json')))
    m1 = json_image_patch_relation1["z"]
    m2 = json_image_patch_relation2["z"]

    d_sum = 0.0

    x1 = len(m1) - 1
    y1 = len(m1[0]) - 1
    x2 = len(m2) - 1
    y2 = len(m2[0]) - 1

    count = 0.0
    for i in range(x1):
        for j in range(y1):

            k_min = -1
            l_min = -1
            min_d = sys.float_info.max
            min_d_eud = sys.float_info.max

            is_consider = 0

            for k in range(x2):
                for l in range(y2):
                    if m1[i][j] < t and -m2[k][l] < t:
                        d = abs(m1[i][j] - m2[k][l])
                        is_consider = 1
                        if min_d > d:
                            min_d = d
                            k_min = k
                            l_min = l
                        elif min_d == d:
                            d_eud = math.sqrt(((i - k) * (i - k)) + ((j - l) * (j - l)))
                            if min_d_eud > d_eud:
                                min_d_eud = d_eud
                                k_min = k
                                l_min = l

                                # print(str(i) + " " + str(k) + " " + str(j) + " " + str(l) + " = " + str(d) + str(
                                #     min_d_eud))

            if is_consider == 1:
                d_sum = d_sum + math.sqrt(((i - k_min) * (i - k_min)) + ((j - l_min) * (j - l_min)))
                count = count + 1.0


    if count == 0:
        # print(image_name1)
        # print(image_name2)
        # print(m1)
        # print(m2)
        return sys.float_info.max

    return d_sum / count


#relation
def closest_mst_relation(image_name1, image_name_relation):
    result_list = {}

    h_z1 = image_name_relation[image_name1]["h_z"]
    v_z1 = image_name_relation[image_name1]["v_z"]

    for image_name2 in image_name_relation.keys():

        h_z2 = image_name_relation[image_name2]["h_z"]
        v_z2 = image_name_relation[image_name2]["v_z"]

        t_h, p_h = stats.ttest_ind(h_z1, h_z2)
        t_v, p_v = stats.ttest_ind(v_z1, v_z2)

        result_list[image_name2] = (p_h+p_v)/2.0

    return result_list


def distance2 (image_name1, image_name2, image_name_relation, t):
    m1 = image_name_relation[image_name1]["z"]
    m2 = image_name_relation[image_name2]["z"]

    d_sum = 0.0

    if is_consider == 1:
        d_sum = d_sum + math.sqrt(((i - k_min) * (i - k_min)) + ((j - l_min) * (j - l_min)))
        count = count + 1.0

def _closest_mst_relation_matrix2(image_name1, image_name2, image_name_relation, t):
    list_of_closest = {}
    # print("------------"+image_name1)
    # print(image_name_relation.keys())
    # print(image_name_relation[image_name1].keys())
    m1 = image_name_relation[image_name1]["z"]
    m2 = image_name_relation[image_name2]["z"]

    d_sum = 0.0

    x1 = len(m1) - 1
    y1 = len(m1[0]) - 1
    x2 = len(m2) - 1
    y2 = len(m2[0]) - 1

    count = 0.0
    for i in range(x1):
        for j in range(y1):

            list_of_closest[str(i)+"_"+str(j)] = []

            k_min = -1
            l_min = -1
            min_d = sys.float_info.max
            min_d_eud = sys.float_info.max

            is_consider = 0

            for k in range(x2):
                for l in range(y2):
                    if m1[i][j] < t and -m2[k][l] < t:
                        d = abs(m1[i][j] - m2[k][l])
                        is_consider = 1
                        if min_d > d:
                            min_d = d
                            k_min = k
                            l_min = l
                        elif min_d == d:
                            d_eud = math.sqrt(((i - k) * (i - k)) + ((j - l) * (j - l)))
                            if min_d_eud > d_eud:
                                min_d_eud = d_eud
                                k_min = k
                                l_min = l

                                # print(str(i) + " " + str(k) + " " + str(j) + " " + str(l) + " = " + str(d) + str(
                                #     min_d_eud))



            if is_consider == 1:
                list_of_closest[str(i) + "_" + str(j)].append(str(k_min)+"_"+str(l_min))
                list_of_closest[str(i) + "_" + str(j)].append(math.sqrt(((i - k_min) * (i - k_min)) + ((j - l_min) * (j - l_min))))

                d_sum = d_sum + math.sqrt(((i - k_min) * (i - k_min)) + ((j - l_min) * (j - l_min)))
                count = count + 1.0


    if count == 0:
        # print(image_name1)
        # print(image_name2)
        # print(m1)
        # print(m2)
        return sys.float_info.max

    return d_sum / count, list_of_closest


def _closest_mst_relation_matrix(image_name1, image_name2, image_name_relation, t):

    # print("------------"+image_name1)
    # print(image_name_relation.keys())
    # print(image_name_relation[image_name1].keys())
    m1 = image_name_relation[image_name1]["z"]
    m2 = image_name_relation[image_name2]["z"]

    d_sum = 0.0

    x1 = len(m1) - 1
    y1 = len(m1[0]) - 1
    x2 = len(m2) - 1
    y2 = len(m2[0]) - 1

    count = 0.0
    for i in range(x1):
        for j in range(y1):

            k_min = -1
            l_min = -1
            min_d = sys.float_info.max
            min_d_eud = sys.float_info.max

            is_consider = 0

            for k in range(x2):
                for l in range(y2):
                    if m1[i][j] < t and -m2[k][l] < t:
                        d = abs(m1[i][j] - m2[k][l])
                        is_consider = 1
                        if min_d > d:
                            min_d = d
                            k_min = k
                            l_min = l
                        elif min_d == d:
                            d_eud = math.sqrt(((i - k) * (i - k)) + ((j - l) * (j - l)))
                            if min_d_eud > d_eud:
                                min_d_eud = d_eud
                                k_min = k
                                l_min = l

                                # print(str(i) + " " + str(k) + " " + str(j) + " " + str(l) + " = " + str(d) + str(
                                #     min_d_eud))

            if is_consider == 1:
                d_sum = d_sum + math.sqrt(((i - k_min) * (i - k_min)) + ((j - l_min) * (j - l_min)))
                count = count + 1.0


    if count == 0:
        # print(image_name1)
        # print(image_name2)
        # print(m1)
        # print(m2)
        return sys.float_info.max

    return d_sum / count

def closest_mst_relation_matrix(image_name1, image_name_relation, t):
    result_list = {}

    # m1 = image_name_relation[image_name1]["m"]

    for image_name2 in image_name_relation.keys():
        print(image_name1 + " --- " + image_name2)
        d1 = _closest_mst_relation_matrix(image_name1, image_name2, image_name_relation, t)
        d2 = _closest_mst_relation_matrix(image_name2, image_name1, image_name_relation, t)

        if d1 < d2:
            result_list[image_name2] = d1
        else:
            result_list[image_name2] = d2

    return result_list



def create_D_patch_matrix(image_path_sorting_list, patch_train_output_reg_folder, t):

    result_D_relation = {}

    for i in range(len(image_path_sorting_list)):
        # print(image_path_sorting_list[i])
        base = os.path.basename(image_path_sorting_list[i])
        image_name = os.path.splitext(base)[0]

        result_D_relation[image_name] = {}
        result_D_relation[image_name]["m"] = []
        result_D_relation[image_name]["m_h"] = []
        result_D_relation[image_name]["h"] = []
        result_D_relation[image_name]["m_v"] = []
        result_D_relation[image_name]["v"] = []
        result_D_relation[image_name]["z"] = []

    for x in range(len(image_path_sorting_list)):

        f_index = os.path.basename(dirname(image_path_sorting_list[x]))
        base = os.path.basename(image_path_sorting_list[x])
        image_name = os.path.splitext(base)[0]
        # print(os.path.join(patch_train_output_reg_folder, "result" + str(f_index) + ".json"))
        json_covmatrix_patch_list1 = json.load(
            open(os.path.join(patch_train_output_reg_folder, "result" + str(f_index) + ".json")))

        # print(json_covmatrix_patch_list1.keys())
        print(image_name)
        C_patchs = json_covmatrix_patch_list1[image_name]
        # C_patchs = patch_cov_json[image_name]
        i = 0
        j = 0
        for key in C_patchs.keys():
            i_j = key.split('_')
            if (int(i_j[0]) > i):
                i = int(i_j[0])
            if (int(i_j[1]) > j):
                j = int(i_j[1])

        m = []
        z = []
        print(i, " ", j)
        for a in range(i):
            m_row = []
            z_row = []
            if a + 1 < i:
                for b in range(j):
                    if b + 1 < j:
                        cov_matrix_patch1 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                        cov_matrix_patch2 = C_patchs[str(a + 2) + "_" + str(b + 1)]

                        d = CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                        d2 = CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)

                        cov_matrix_patch3 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                        cov_matrix_patch4 = C_patchs[str(a + 1) + "_" + str(b + 2)]

                        d3 = CovarianceDistance(cov_matrix_patch3, cov_matrix_patch4)
                        d4 = CovarianceDistance(cov_matrix_patch4, cov_matrix_patch3)

                        if (d > d2):
                            d = d2
                        if d > t:
                            d = t
                        m_row.append(d)
                        result_D_relation[image_name]["h"].append(d)

                        if (d3 > d4):
                            d3 = d4
                        if d3 > t:
                            d3 = t
                        z_row.append((d3 + d) / 2.0)
                        result_D_relation[image_name]["z"].append((d3 + d) / 2.0)

            if (len(m_row) != 0):
                m.append(m_row)
            if (len(z_row) != 0):
                z.append(z_row)

        result_D_relation[image_name]["z"] = z
        result_D_relation[image_name]["m"] = m
        result_D_relation[image_name]["m_h"] = m

        m_v = []
        for a in range(j):
            m_row = []
            if a + 1 < j:
                for b in range(i):
                    if b + 1 < i:
                        cov_matrix_patch1 = C_patchs[str(b + 1) + "_" + str(a + 1)]
                        cov_matrix_patch2 = C_patchs[str(b + 1) + "_" + str(a + 2)]

                        d = CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                        d2 = CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)
                        if (d > d2):
                            d = d2
                        if d > t:
                            d = t
                        m_row.append(d)
                        result_D_relation[image_name]["v"].append(d)

            if (len(m_row) != 0):
                m_v.append(m_row)
        result_D_relation[image_name]["m_v"] = m_v

    for name in result_D_relation.keys():
        h_z = stats.zscore(np.array(result_D_relation[name]["h"]))
        result_D_relation[name]["h_z"] = h_z.tolist()
        v_z = stats.zscore(np.array(result_D_relation[name]["v"]))
        result_D_relation[name]["v_z"] = v_z.tolist()

    return result_D_relation





#-----------------



#t is theshold of accepted it similar, more than this not considering
def graph_image_patch(nx_graph, image_name1, image_name_sorting_list, json_covmatrix_patch_list, t, dbscan_t):

    C1_patchs = json_covmatrix_patch_list[image_name1]
    for i in range(len(image_name_sorting_list)):
        image_name2 = image_name_sorting_list[i]

        if image_name1 != image_name2:

            nx_graph.add_node(image_name1)
            nx_graph.add_node(image_name2)

            C2_patchs = json_covmatrix_patch_list[image_name2]
            similarity_mean = hcompare_patch_similarity(C1_patchs, C2_patchs, t)
            # print(image_name1 + " " + image_name2 + " " + str(similarity_mean))
            # similarity_mean2 = hcompare_patch_similarity(C2_patchs, C1_patchs, t)
            # print(image_name2 + " " + image_name1 + " " + " " + str(similarity_mean2))
            # print("---")
            if similarity_mean <= dbscan_t:
                nx_graph.add_edge(image_name1, image_name2, weight=similarity_mean)



















