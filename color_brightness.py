# coding=utf-8
import os
import os
import cv2
import math
import numpy as np
import shutil
from PIL import Image
from PIL import ImageEnhance

"""
1、对比度：白色画面(最亮时)下的亮度除以黑色画面(最暗时)下的亮度；
2、色彩饱和度：：彩度除以明度，指色彩的鲜艳程度，也称色彩的纯度；
3、色调：向负方向调节会显现红色，正方向调节则增加黄色。适合对肤色对象进行微调；
4、锐度：是反映图像平面清晰度和图像边缘锐利程度的一个指标。
"""


def compute(img):
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Bmean.append(np.mean(img[:, :, 0]))
    per_image_Gmean.append(np.mean(img[:, :, 1]))
    per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return math.sqrt(0.241 * (R_mean ** 2) + 0.691 * (G_mean ** 2) + 0.068 * (B_mean ** 2))


def fun_color(image, coefficient, path_save):
    # 色度,增强因子为1.0是原始图像
    # 色度增强 1.5
    # 色度减弱 0.8
    enh_col = ImageEnhance.Color(image)
    image_colored1 = enh_col.enhance(coefficient)
    image_colored1.save(path_save)


def fun_Contrast(image, coefficient, path_save):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强 1.5
    # 对比度减弱 0.8
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted1 = enh_con.enhance(coefficient)
    image_contrasted1.save(path_save)

def fun_Sharpness(image, coefficient, path_save):
    # 锐度，增强因子为1.0是原始图片
    # 锐度增强 3
    # 锐度减弱 0.8
    enh_sha = ImageEnhance.Sharpness(image)
    image_sharped1 = enh_sha.enhance(coefficient)
    image_sharped1.save(path_save)

def fun_bright(image, coefficient, path_save):
    # 变亮 1.5
    # 变暗 0.8
    # 亮度增强,增强因子为0.0将产生黑色图像； 为1.0将保持原始图像。
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened1 = enh_bri.enhance(coefficient)
    image_brightened1.save(path_save)

def show_all():
    file_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/JPEGImages/"
    xml_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/Annotations/"
    save_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/save/"
    xml_save = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/xml/"
    list_file = os.listdir(file_root)
    cnt = 0
    for img_name in list_file:
        cnt += 1
        print("cnt=%d,img_name=%s" % (cnt, img_name))
        path = file_root + img_name
        name = img_name.replace(".jpg", "")
        image = Image.open(path)
        #xml_path = 
        list_coe = [0.5,1,3]
        for val in list_coe:
            path_save_bright = save_root + name + "_bri_" + str(val) + ".jpg"
            fun_bright(image, val, path_save_bright)

            path_save_color = save_root + name + "_color_" + str(val) + ".jpg"
            fun_color(image, val, path_save_color)

            path_save_contra = save_root + name + "_contra_" + str(val) + ".jpg"
            fun_Contrast(image, val, path_save_contra)

            path_save_sharp = save_root + name + "_sharp_" + str(val) + ".jpg"
            fun_Sharpness(image, val, path_save_sharp)


def my_aug():
    file_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/JPEGImages/"
    save_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/save/"
    xml_save = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/xml/"
    xml_root = "/home/zhangwanchun/data/VOCdevkit/VOC2007_org/Annotations/"
    list_file = os.listdir(file_root)
    cnt = 0
    for img_name in list_file:
        cnt += 1
        print("cnt=%d,img_name=%s" % (cnt, img_name))
        path = file_root + img_name
        name = img_name.replace(".jpg", "")
        image = Image.open(path)
        img = cv2.imread(path)
        mean_1 = compute(img)
        cof = 0.0
        if mean_1 < 40:
            cof = 3.5
        elif mean_1 < 60:
            cof = 3
        elif mean_1 < 80:
            cof = 2
        elif mean_1 < 90:
            cof = 1.5
        elif mean_1 < 110:
            cof = 1.1
        elif mean_1 > 130:
             cof = 0.5
        else:
             cof = 0.75

        cof_contrast = 0.0
        if cof>1:
            cof_contrast = 1.5
        else:
            cof_contrast = 0.8
        xmlpath = xml_root + name + '.xml'
        
        path_save_bright = save_root + name + "_bri_" + str(cof) + '.jpg'
        path_save_bright_xml = xml_save + name + "_bri_" + str(cof) + '.jpg'
        shutil.copy(xmlpath,path_save_bright_xml.replace(".jpg", ".xml"))
        fun_bright(image, cof, path_save_bright)

        path_save_sharp = save_root + name + "_sharp_" + str(2) + '.jpg'
        path_save_sharp_xml = xml_save + name + "_sharp_" + str(2) + '.jpg'
        shutil.copy(xmlpath, path_save_sharp_xml.replace(".jpg", ".xml"))
        fun_Sharpness(image, 2, path_save_sharp)

        path_save_contra = save_root + name + "_contra_" + str(cof_contrast) + ".jpg"
        path_save_contra_xml = xml_save + name + "_contra_" + str(cof_contrast) + ".jpg"
        shutil.copy(xmlpath, path_save_contra_xml.replace(".jpg", ".xml"))
        fun_Contrast(image, cof_contrast, path_save_contra)

        path_save_color = save_root + name + "_color_" + str(1.5) + ".jpg"
        path_save_color_xml = xml_save + name + "_color_" + str(1.5) + ".jpg"
        shutil.copy(xmlpath,path_save_color_xml.replace(".jpg", ".xml"))
        fun_color(image, 1.5, path_save_color)


if __name__ == "__main__":

    #show_all()
    my_aug()
