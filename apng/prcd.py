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

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure



from scipy import stats

np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')








#paramenter and data tempolary path
_patch_size = 16
_max_similar_t = 50
_patch_val_16 = "C:\\apng\\venv\\patch_16\\"
_tmp_resize = "C:\\apng\\venv\\_tmp_resize.png"
_tmp_result_D_relation_train = "C:\\apng\\venv\\_tmp_resize.png\\tmp_result_D_relation_train"
_illus = "C:\\apng\\venv\\_illus\\"





descriptor_relation = {}

def img_patch_to_cov(patch_folder):

    files = get_file(patch_folder)
    json_result = {}
    for img_file in files:
        i_j_f = img_file.split('_')
        f = i_j_f[2].split('.')[0]

        img_path = os.path.join(patch_folder, img_file)
        image = cv2.imread(img_path)
        resized = cv2.resize(image, (_patch_size, _patch_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('tmp.png', resized)
        img = Image.open('tmp.png')
        # C1 = reg_cov.RegionCovarianceDescriptor(reg_cov.FeatureImage2(img)) #rgb
        C1 = reg_cov.RegionCovarianceDescriptor(reg_cov.FeatureImage3(img))  # gray

        if f in json_result.keys():
            x = 0
        else:
            json_result[f] = {}

        json_result[f][i_j_f[0] + "_" + i_j_f[1]] = C1.tolist()

    return json_result

def relation_descriptor(patch_cov_json, t):
    result_D_relation = {}
    # print(patch_cov_json.keys())
    for key in patch_cov_json.keys():
        result_D_relation[key] = {}
        result_D_relation[key]["m"] = []
        result_D_relation[key]["m_h"] = []
        result_D_relation[key]["h"] = []
        result_D_relation[key]["m_v"] = []
        result_D_relation[key]["v"] = []
        result_D_relation[key]["z"] = []

    for key in patch_cov_json.keys():

        image_name = key

        C_patchs = patch_cov_json[image_name]
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
        for a in range(i):
            m_row = []
            z_row = []
            if a + 1 < i:
                for b in range(j):
                    if b + 1 < j:

                        cov_matrix_patch1 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                        cov_matrix_patch2 = C_patchs[str(a + 2) + "_" + str(b + 1)]

                        d = reg_cov.CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                        d2 = reg_cov.CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)


                        cov_matrix_patch3 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                        cov_matrix_patch4 = C_patchs[str(a + 1) + "_" + str(b + 2)]

                        d3 = reg_cov.CovarianceDistance(cov_matrix_patch3, cov_matrix_patch4)
                        d4 = reg_cov.CovarianceDistance(cov_matrix_patch4, cov_matrix_patch3)


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

            if(len(m_row) != 0):
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

                        d = reg_cov.CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                        d2 = reg_cov.CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)

                        if (d > d2):
                            d = d2
                        if d > t:
                            d = t
                        m_row.append(d)
                        result_D_relation[image_name]["v"].append(d)

            if (len(m_row) != 0):
                m_v.append(m_row)

        result_D_relation[image_name]["m_v"] = m_v
    # for name in result_D_relation.keys():
    #     h_z = stats.zscore(np.array(result_D_relation[name]["h"]))
    #     result_D_relation[name]["h_z"] = h_z.tolist()
    #     v_z = stats.zscore(np.array(result_D_relation[name]["v"]))
    #     result_D_relation[name]["v_z"] = v_z.tolist()

    return result_D_relation

def json_descriptor(name, path, descriptor_relation):

    if os.path.exists(_patch_val_16):
        try:
            shutil.rmtree(_patch_val_16)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    os.mkdir(os.path.join(_patch_val_16))

    base = os.path.basename(os.path.join(path))
    image1 = os.path.splitext(base)[0]


    patch(image1+".jpg", path, _patch_size, os.path.join(_patch_val_16))
    patch_cov = img_patch_to_cov(_patch_val_16)
    descriptor_relation1 = relation_descriptor(patch_cov, _max_similar_t)
    descriptor_relation[name] = descriptor_relation1[image1]

    return descriptor_relation

#prcdv2
def patch_stride(img_name_ext, img_path, window_size, output_folder):
    imgx = Image.open(img_path)
    img = cv2.imread(img_path)
    img_w, img_h, c = img.shape

    w_num = math.floor(img_w / (window_size * 1.0))
    h_num = math.floor(img_h / (window_size * 1.0))

    for i in range(int(w_num)):
        for j in range(int(h_num)):
            img_crop = imgx.crop(
                (j * window_size, i * window_size, (j + 1) * window_size, (i + 1) * window_size))
            img_crop.save(os.path.join(output_folder, str(i + 1) + '_' + str(j + 1) + '_' + img_name_ext))



def create_D_patch_matrix_coco2(image_val_name, patch_train_16_patch_json, t, out_path):

    result_D_relation = {}
    image_name = image_val_name


    result_D_relation[image_name] = {}
    result_D_relation[image_name]["m"] = []
    result_D_relation[image_name]["m_h"] = []
    result_D_relation[image_name]["h"] = []
    result_D_relation[image_name]["m_v"] = []
    result_D_relation[image_name]["v"] = []
    result_D_relation[image_name]["z"] = []
    result_D_relation[image_name]["ok"] = 0

    count = 0
    for image_name in result_D_relation.keys():
        if image_name not in result_D_relation.keys() or result_D_relation[image_name]["ok"] != 1 :
            result_D_relation[image_name]["ok"] = 1
            if not os.path.exists(os.path.join(out_path,image_name + ".json")):

                json_covmatrix_patch_list1 = json.load(
                    open(os.path.join(patch_train_16_patch_json, image_name + ".json")))


                #print("create_D_patch_matrix_coco2", json_covmatrix_patch_list1.keys())
                C_patchs = json_covmatrix_patch_list1[image_name]

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
                for a in range(i):
                    m_row = []
                    z_row = []
                    if a + 1 < i:
                        for b in range(j):
                            if b + 1 < j:
                                cov_matrix_patch1 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                                cov_matrix_patch2 = C_patchs[str(a + 2) + "_" + str(b + 1)]

                                # print('------')
                                # print(cov_matrix_patch1)
                                # print(cov_matrix_patch2)
                                # print('------')

                                d = reg_cov.CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                                d2 = reg_cov.CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)

                                cov_matrix_patch3 = C_patchs[str(a + 1) + "_" + str(b + 1)]
                                cov_matrix_patch4 = C_patchs[str(a + 1) + "_" + str(b + 2)]

                                d3 = reg_cov.CovarianceDistance(cov_matrix_patch3, cov_matrix_patch4)
                                d4 = reg_cov.CovarianceDistance(cov_matrix_patch4, cov_matrix_patch3)
                                # print(d,d2,d3,d4)

                                if (d > d2):
                                    d = d2
                                if d > t:
                                    d = t
                                m_row.append(d)

                                # if (image_name == 'tmpb2'):
                                #     print(d)

                                result_D_relation[image_name]["h"].append(d)

                                if (d3 > d4):
                                    d3 = d4
                                if d3 > t:
                                    d3 = t

                                if (((d3 + d) / 2.0) == float("inf")):
                                    z_row.append(t)
                                    result_D_relation[image_name]["z"].append(t)
                                else:
                                    z_row.append((d3 + d) / 2.0)
                                    result_D_relation[image_name]["z"].append((d3 + d) / 2.0)
                    # else:
                    #     if (image_name == 'tmpb2'):
                    #         print('1234', range(i))

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

                                d = reg_cov.CovarianceDistance(cov_matrix_patch1, cov_matrix_patch2)
                                d2 = reg_cov.CovarianceDistance(cov_matrix_patch2, cov_matrix_patch1)
                                if (d > d2):
                                    d = d2
                                if d > t:
                                    print(d)
                                    d = t
                                m_row.append(d)
                                result_D_relation[image_name]["v"].append(d)

                    if (len(m_row) != 0):
                        m_v.append(m_row)
                result_D_relation[image_name]["m_v"] = m_v

                f = open(os.path.join(out_path, image_name + '.json'), 'w')
                f.write(json.dumps(result_D_relation[image_name]))
                f.close()



        count = count + 1

    return result_D_relation

def avg(lst):
    return sum(lst)/len(lst)

def hist_n(image_1, bin):
    img1 = cv2.imread(image_1)

    # Convert it to HSV
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    hist_r = cv2.calcHist([img1_hsv], [0], None, [bin], [0.0, 255.0]).tolist()
    hist_g = cv2.calcHist([img1_hsv], [1], None, [bin], [0.0, 255.0]).tolist()
    hist_b = cv2.calcHist([img1_hsv], [2], None, [bin], [0.0, 255.0]).tolist()

    # print(hist_r)
    return hist_r, hist_g, hist_b

def encode_rgb(r_avg):
    r_median = []
    tmp_median = []
    print(len(r_avg))
    count = 0
    for x1 in range(len(r_avg)):
        if 2 == count:
            r_median.append(statistics.median(tmp_median))
            tmp_median = []
            count = 0
        else:
            tmp_median.append(r_avg[x1])
        count = count + 1
    r_avgx = r_median
    return r_avgx

def encode_median(array, encode_size, min_sim_value, json_covarince_patch):


    max_i = 0
    max_j = 0
    for key_patch in json_covarince_patch.keys():
        i1 = int(key_patch.split('_')[0])
        if i1 > max_i:
            max_i = i1

        j1 = int(key_patch.split('_')[1])
        if j1 > max_j:
            max_j = j1

    h = []
    for x in range(max_i):
        for y in range(max_j):
            h.append(json_covarince_patch[str(x+1)+"_"+str(y+1)])

    v = []
    for y in range(max_j):
        for x in range(max_i):
            v.append(json_covarince_patch[str(x+1) + "_" + str(y+1)])




    new_array = []
    for a in array:
        for b in a:
            if b >= min_sim_value:
                new_array.append(min_sim_value)
            else:
                new_array.append(b)
    # block_num = math.floor(len(new_array) / ((size_encode) * 1.0))
    # print("block_num",block_num)

    encode = []

    median = []
    for i in range(len(new_array)):
        if (((i+1) % int(encode_size)) != 0):
            if (new_array[i] > min_sim_value):
                median.append(min_sim_value)
            else:
                median.append(new_array[i])
        else:
            encode.append(statistics.median(median))

            median = []

            if (new_array[i] > min_sim_value):
                median.append(min_sim_value)
            else:
                median.append(new_array[i])



def encoder_prcdv2(window_size, block_num, img_path, img_name, encode_size):
    resize_wh = window_size * (block_num + 1)

    _tmp_val = os.path.join("C:\\apng\\venv\\_tmp_val")
    if not os.path.exists(_tmp_val):
        os.mkdir(os.path.join(_tmp_val))

    imgx = Image.open(os.path.join(img_path, img_name + ".jpg"))
    dsize = (resize_wh, resize_wh)
    imgx = imgx.resize(dsize, Image.ANTIALIAS)
    imgx.save(os.path.join(_tmp_val, img_name + ".png"), "png")
    img1 = cv2.imread(os.path.join(_tmp_val, img_name + ".png"))
    img_w, img_h, c = img1.shape
    print(img_w, img_h, c)

    image_subx = img_name + ".png"

    image_sub = os.path.splitext(image_subx)[0]

    patch_tmp_16 = _patch_val_16
    if os.path.exists(patch_tmp_16):
        try:
            shutil.rmtree(patch_tmp_16)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    os.mkdir(os.path.join(patch_tmp_16))

    # print(os.path.join(_tmp2, image_subx))
    patch_stride(image_sub + ".png", os.path.join(_tmp_val, image_subx), window_size, patch_tmp_16)

    tmp_patch_json = _patch_val_16
    if os.path.exists(tmp_patch_json):
        try:
            shutil.rmtree(tmp_patch_json)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    os.mkdir(os.path.join(tmp_patch_json))

    files2 = get_file(patch_tmp_16)
    json_result = {}
    r_avg = []
    g_avg = []
    b_avg = []
    # print(">>>>>>",len(files2))
    for img_file2 in files2:
        i_j_f = img_file2.split('_')
        f = image_sub
        image = cv2.imread(os.path.join(patch_tmp_16, img_file2))

        resized = cv2.resize(image, (window_size, window_size), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(_tmp_resize, resized)

        #hist rgb
        hist_r, hist_g, hist_b = hist_n(_tmp_resize, 16)

        r = []
        g = []
        b = []
        for j in range(len(hist_r)):
            r.append(hist_r[j][0])
            g.append(hist_g[j][0])
            b.append(hist_b[j][0])

        r_avg.append(avg(r))
        g_avg.append(avg(g))
        b_avg.append(avg(b))

        ####


        img = Image.open(_tmp_resize)

        C1 = reg_cov.RegionCovarianceDescriptor(reg_cov.FeatureImage2(img))
        if f in json_result.keys():
            x = 0
        else:
            json_result[f] = {}

        if i_j_f[0] == "0" or i_j_f[1] == "0":
            x = 0
        else:
            json_result[f].update({i_j_f[0] + "_" + i_j_f[1]: C1.tolist()})

    # r_avg = encode_rgb(r_avg)
    # g_avg = encode_rgb(g_avg)
    # b_avg = encode_rgb(b_avg)


    # print('end 1. resize and patch covariance')
    # print(json_result)
    f = open(os.path.join(tmp_patch_json, image_sub + ".json"), 'w')
    f.write(json.dumps(json_result))
    f.close()
    # print('end 2. resize and patch covariance')

    tmp_result_D_relation_train = _tmp_result_D_relation_train
    if os.path.exists(tmp_result_D_relation_train):
        try:
            shutil.rmtree(tmp_result_D_relation_train)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    os.mkdir(os.path.join(tmp_result_D_relation_train))

    reg_cov._closest_mst_relation_matrix("a1", "a2", descriptor_relation, _max_similar_t)

    result_D_relation = create_D_patch_matrix_coco2(image_sub, tmp_patch_json, sys.float_info.max,
                                                    tmp_result_D_relation_train)


    h = result_D_relation[image_sub]['m_h']
    v = result_D_relation[image_sub]['m_v']
    encode_h, block_num = \
        encode_median(result_D_relation[image_sub]['m_h'],
                      encode_size, 50, json_result[image_sub])

    encode_v, block_num = \
        encode_median(result_D_relation[image_sub]['m_v'],
                      encode_size, 50, json_result[image_sub])

    return encode_h, encode_v ,h ,v, r_avg, g_avg, b_avg

#prcdv2
def img_resize(img_path, img_name):
    resize_wh = 496

    _tmp_val = os.path.join("C:\\apng\\venv\\resize\\")
    if not os.path.exists(_tmp_val):
        os.mkdir(os.path.join(_tmp_val))

    imgx = Image.open(os.path.join(img_path, img_name + ".jpg"))
    dsize = (resize_wh, int(resize_wh/2))
    imgx = imgx.resize(dsize, Image.ANTIALIAS)
    imgx.save(os.path.join(_tmp_val, img_name + ".png"), "png")
    return os.path.join(_tmp_val, img_name + ".png")

def img_resize2(img_path, img_name):
    resize_wh = 496

    _tmp_val = os.path.join("C:\\apng\\venv\\resize\\")
    if not os.path.exists(_tmp_val):
        os.mkdir(os.path.join(_tmp_val))

    imgx = Image.open(os.path.join(img_path, img_name))
    dsize = (resize_wh,resize_wh)
    imgx = imgx.resize(dsize, Image.ANTIALIAS)
    imgx.save(os.path.join(_tmp_val, img_name), "png")
    return os.path.join(_tmp_val, img_name)



def distance3(hog1, hog2):
    sum1 = 0.0
    print("HOG shape", len(hog1))
    for i in range(len(hog1)):
        sum1 = sum1 + abs(hog1[i]-hog2[i])

    return sum1

def distance2(prcd1, prcd2):

    sum1 = 0.0
    print("prcdv2 shape", len(prcd1["z"]))
    for i in range(len(prcd1["z"])):
        for j in range(len(prcd1["z"][i])):
            sum1 = sum1 + abs(prcd1["z"][i][j] - prcd2["z"][i][j])

    return sum1


def coil100_experiment():
    _coil_100 = "C:\\apng\\coil_100\\"
    gray_descriptor = "C:\\apng\\gray_descriptor"
    _label_init = "obj"
    _tmpxxx = "C:\\apng\\tmpxxx.png"


    file_names = get_file(_coil_100)

    label = {}
    for file in file_names:
        l = file.split("__")[0]
        if l not in label:
            label[l] = []

            path = _coil_100
            files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and \
                     l+"__" in i]

            number = []
            for file in files:
                number.append(int(file.split("__")[1].split(".png")[0]))
                number_sort = sorted(number)

            for n in number_sort:
                label[l].append( os.path.join(_coil_100, l+"__"+str(n)+".png") )








    # descriptor
    print("--descriptor generate--")
    for key in label.keys():

        descriptor_relation1 = {}
        descriptor_relation2 = {}
        descriptor_relation3 = {}

        for src in label[key]:


            print(key, src)
            base = os.path.basename(os.path.join(src))
            image1 = os.path.splitext(base)[0]
            if not os.path.exists(os.path.join(gray_descriptor, image1 + ".json")):
                # im = Image.open(src)
                # im.save(_tmpxxx)
                #
                # json_descriptor(image1, _tmpxxx, descriptor_relation1)

                img = img_resize2(_coil_100, base)
                im = Image.open(img)
                im.save(_tmpxxx)

                json_descriptor(image1, _tmpxxx, descriptor_relation2)

                _tmpxxx2 = "C:\\apng\\tmpxxx2.png"
                image = cv2.imread(_tmpxxx)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(os.path.join(_tmpxxx2), backtorgb)



                fd, hog_image = hog_descriptor(_tmpxxx2)
                hog_descriptor1 = np.concatenate(np.array(hog_image)).tolist()
                descriptor_relation3[image1] = hog_descriptor1

                f = open(os.path.join(gray_descriptor, image1 + ".json"), 'w')
                f.write(json.dumps({"hog": descriptor_relation3[image1], "prcdv2": descriptor_relation2[image1]}))
                f.close()


def nDCG(rank_data):
    max_sum = 0
    for i in range(1, len(rank_data)):
        rel = (len(rank_data)-1) - i
        max_sum = max_sum + (rel / math.log(i + 1, 2))

    score = 0
    i = 1
    for data in rank_data:
        if data != 0:
            rel = (len(rank_data)-1) - data
            score = score + (rel / math.log(i + 1, 2))
            i = i + 1

    return score/max_sum, score, max_sum




def coil100_experiment_measurement():
    _coil_100 = "C:\\apng\\coil_100\\"
    _coil_100_result = "C:\\apng\\coil_100_result\\"
    gray_descriptor = "C:\\apng\\gray_descriptor"

    file_names = get_file(_coil_100)

    label = {}
    for file in file_names:
        l = file.split("__")[0]
        if l not in label:
            label[l] = []

            path = _coil_100
            files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and \
                     l + "__" in i]

            number = []
            for file in files:
                number.append(int(file.split("__")[1].split(".png")[0]))
                number_sort = sorted(number)

            for n in number_sort:
                label[l].append(l + "__" + str(n))




    n = 0.0
    speed1 = []
    speed2 = []


    i = 0
    descriptor_relation1 = {}
    descriptor_relation2 = {}
    descriptor_relation3 = {}

    result0 = []
    result1 = []
    result2 = []
    result3 = []
    k = list(label.keys())
    s1 = 0.0
    s2 = 0.0
    count_test = 0

    current_label = ""
    input = ""
    for l in k:
        for key in label[l]:
            src = key

            # measurement
            print("--measurement--")

            if i == 0 or input == "" or input.split("__")[0] != src.split("__")[0]:
                print(input)
                dataset = []
                input = src
                result0 = []
                result1 = []
                result2 = []
                result3 = []
                descriptor_relation1 = {}
                descriptor_relation2 = {}
                descriptor_relation3 = {}
                s1 = 0.0
                s2 = 0.0


            dataset.append(src)

            print(input, src)



            if input not in descriptor_relation2.keys():
                descriptor_relation2[input] = json.load(open(os.path.join(gray_descriptor, input + ".json")))
                descriptor_relation3[input] = json.load(open(os.path.join(gray_descriptor, input + ".json")))


            if src not in descriptor_relation2.keys():
                descriptor_relation2[src] = json.load(open(os.path.join(gray_descriptor, src + ".json")))
                descriptor_relation3[src] = json.load(open(os.path.join(gray_descriptor, src + ".json")))

            # d1 = reg_cov._closest_mst_relation_matrix(input, src, descriptor_relation1, _max_similar_t)
            # result1[input].append(d1)

            result0.append(src)

            start_time = time.time()
            d2 = distance2(descriptor_relation2[input]["prcdv2"], descriptor_relation2[src]["prcdv2"])
            result2.append(d2)
            s1 = s1 + (time.time() - start_time)

            start_time = time.time()
            d3 = distance3(descriptor_relation3[input]["hog"], descriptor_relation3[src]["hog"])
            result3.append(d3)
            s2 = s2 + (time.time() - start_time)

            # print("PRCDv2 size", descriptor_relation2[input]["prcdv2"], descriptor_relation2[src]["prcdv2"])

            n = n+1




            i = i + 1

            if i == 11 or input.split("__")[0] != src.split("__")[0]:
                i = 0

                speed1.append(s1)
                speed2.append(s2)

                print("PRCD2 gray 496x496", s1)
                print("HOG gray 496x496", s2)


                f = open( os.path.join(_coil_100_result, input+".json"), 'w')
                f.write(json.dumps({"dataset": dataset, "result0": result0, "result1": result1, "result2": result2, "result3": result3}))
                f.close()
                if count_test == 400:
                    break
                else:
                    count_test = count_test + 1

    f = open(os.path.join("speed1.json"), 'w')
    f.write(json.dumps(speed1))
    f.close()

    f = open(os.path.join("speed2.json"), 'w')
    f.write(json.dumps(speed2))
    f.close()

    y1 = np.array(speed1)
    y2 = np.array(speed2)

    print("avg PRCDv2 descriptor (Grayscale)", _sum(speed1)/n)
    print("avg HOG descriptor (Grayscale)", _sum(speed2)/n)


    # plt.plot(y1, color="blue",linewidth=1.0)
    # plt.plot(y2, color="r",linewidth=1.0, linestyle='--')
    #
    # plt.legend(('PRCDv2 496x496 (Grayscale)', 'HOG 496x496 (Grayscale)'),
    #            loc='upper right')
    #
    # plt.grid()
    # font1 = {'family': 'serif','size': 12}
    #
    # plt.xlabel("number of tests (each test has 10 descriptor)", fontdict = font1)
    # plt.ylabel("Average comparison test descriptors", fontdict = font1)
    #
    # plt.show()


def _sum(arr):
    # initialize a variable
    # to store the sum
    # while iterating through
    # the array later
    sum = 0

    # iterate through the array
    # and add each element to the sum variable
    # one at a time
    for i in arr:
        sum = sum + i

    return (sum)


def coil100_experiment_DCG():

    _coil_100_result = "C:\\apng\\coil_100_result\\"
    coil_100_result_sort = "C:\\apng\\coil_100_result_sort\\"
    files = get_file(_coil_100_result)


    prcdv2_avg_ndcg = 0
    hog_avg_ndcg = 0

    sample = 0
    for file in files:
        json_result = json.load(
            open(os.path.join(_coil_100_result, file)))

        if len(json_result["result2"]) == 11:
            sort_d2 = list(numpy.argsort(json_result["result2"]))
            sort_d3 = list(numpy.argsort(json_result["result3"]))

            nDCG_prcdv2, prcdv2_score, prcdv2_max_sum = nDCG(sort_d2)
            nDCG_hog, hog_score, hog_max_sum = nDCG(sort_d3)

            print("-----> "+file, json_result["result0"])
            print("prcdv2", nDCG_prcdv2, sort_d2)
            print("hog   ", nDCG_hog, sort_d3)

            s2 = []
            for d2 in sort_d2:
                s2.append(int(str(d2)))

            s3 = []
            for d3 in sort_d3:
                s3.append(int(str(d3)))


            result = [s2, s3]

            prcdv2_avg_ndcg = prcdv2_avg_ndcg + nDCG_prcdv2
            hog_avg_ndcg = hog_avg_ndcg + nDCG_hog

            f = open(os.path.join(coil_100_result_sort, file + ".json"), 'w')
            f.write(json.dumps(result))
            f.close()

            sample = sample + 1

    print("sample=", sample, "prcdv2 avg=", prcdv2_avg_ndcg/sample, "hog avg avg=", hog_avg_ndcg/sample)




def hog_descriptor(src):
    imgx = Image.open(os.path.join(src))
    fd, hog_image = hog(imgx, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(16, 16), visualize=True)
    return fd, hog_image


def prcd_illustrate(image, prcd, _max_similar_t):
    base = os.path.basename(image)
    image1 = os.path.splitext(base)[0]
    if not os.path.exists(_illus):
        os.mkdir(_illus)

    norm = []

    for i in prcd:
        norm_row = []
        for j in i:

            if j > _max_similar_t:
                a = _max_similar_t
            else:
                a = j
            x = 255.0 - (a * 255.0 / _max_similar_t)
            norm_row.append(int(x))
        norm.append(norm_row)

    data = np.asarray(norm, dtype=np.uint8)
    # # print(data)
    #
    img = Image.fromarray(data).save(os.path.join(_illus, image1 + ".jpg"))


def img_to_gray_scale(_cropx):
    for file in get_file(_cropx):
        if not os.path.exists(_cropx_gray):
            os.mkdir(os.path.join(_cropx_gray))


        base = os.path.basename(os.path.join(_cropx, file))
        image1 = os.path.splitext(base)[0]

        image = cv2.imread(os.path.join(_cropx, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(_cropx_gray, image1 + ".jpg"), backtorgb)





# coil100_experiment()
# coil100_experiment_measurement()
# coil100_experiment_DCG()
# exit(0)







# 1. preprocess image gray scale
print("1. preprocess image gray scale")
_cropx_gray = 'C:\\apng\\_cropx_gray\\'
img_to_gray_scale('C:\\apng\\car_crop\\')

# 2. descritor generate
print("2.descritor generate")
# src='C:\\apng\\car_crop\\' # rgb comparison
src = 'C:\\apng\\_cropx_gray\\' # grayscale comparison

excution_time = []

start_time = time.time()
src_no = '1'
json_descriptor("car1x", src+src_no+".jpg", descriptor_relation)
img = img_resize(src, src_no)
json_descriptor("car1", img, descriptor_relation)
prcd_illustrate(src+src_no+".jpg", descriptor_relation["car1"]["z"], _max_similar_t) #keep PRCD illustration path (_illus = "C:\\apng\\venv\\_illus\\")
excution_time.append(time.time() - start_time)


start_time = time.time()
src_no = '2'
json_descriptor("car2x", src+src_no+".jpg", descriptor_relation)
img = img_resize(src, src_no)
json_descriptor("car2", img, descriptor_relation)
excution_time.append(time.time() - start_time)


start_time = time.time()
src_no = '3'
json_descriptor("car3x", src+src_no+".jpg", descriptor_relation)
img = img_resize(src, src_no)
json_descriptor("car3", img, descriptor_relation)
excution_time.append(time.time() - start_time)


start_time = time.time()
src_no = '4'
json_descriptor("car4x", src+src_no+".jpg", descriptor_relation)
img = img_resize(src, src_no)
json_descriptor("car4", img, descriptor_relation)
excution_time.append(time.time() - start_time)

start_time = time.time()
src_no = '5'
json_descriptor("car5x", src+src_no+".jpg", descriptor_relation)
img = img_resize(src, src_no)
json_descriptor("car5", img, descriptor_relation)
excution_time.append(time.time() - start_time)
start_time = time.time()

# 3.distance between image v1 is slow and v2 (distance2) is faster.
# Process is slow at reg_cov._closest_mst_relation_matrix.
print("3.similarity distance between image")
start_time = time.time()
d1 = reg_cov._closest_mst_relation_matrix("car1x", "car2x", descriptor_relation, _max_similar_t)
print("Distance of car2 = ", d1, (time.time() - start_time), "d2=", distance2(descriptor_relation["car1"], descriptor_relation["car2"]))
start_time = time.time()
d1 = reg_cov._closest_mst_relation_matrix("car1x", "car3x", descriptor_relation, _max_similar_t)
print("Distance of car3 = ", d1, (time.time() - start_time), "d2=", distance2(descriptor_relation["car1"], descriptor_relation["car3"]))
start_time = time.time()
d1 = reg_cov._closest_mst_relation_matrix("car1x", "car4x", descriptor_relation, _max_similar_t)
print("Distance of car4 = d1 = ", d1, (time.time() - start_time), "d2=", distance2(descriptor_relation["car1"], descriptor_relation["car4"]))
start_time = time.time()
d1 = reg_cov._closest_mst_relation_matrix("car1x", "car5x", descriptor_relation, _max_similar_t)
print("Distance of car5 = d1 = ", d1, (time.time() - start_time), "d2=", distance2(descriptor_relation["car1"], descriptor_relation["car5"]))
