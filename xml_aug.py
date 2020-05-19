from xml.dom.minidom import parse
import matplotlib.pyplot as plt
import xml.dom.minidom
import os,shutil
import matplotlib  
import numpy as np
import cv2
from PIL import Image, ImageDraw
##########################################################
root="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/"
#only need to change these
##########################################################
#annroot=root+'2/'
#picroot=root+'1/'
#annroot=root+'xml/'
#picroot=root+'img/'
annroot=root+'Annotations/'
picroot=root+'JPEGImages/'
anns=os.listdir(annroot)
imgs=os.listdir(picroot)
 
labelmap=["cpls"]
 
colormap=["red" , "green", "blue" , "yellow", "pink" , "olive" , "deeppink" , "darkorange", "purple", "cyan","red" , "green", "blue" , "yellow", "pink" , "olive" , "deeppink" , "darkorange", "purple", "cyan","red"]
 
def mkdir(path): 
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
 
number = 0
nn=0
for ann in anns:
    number += 1
    print (number)
    print (ann)
    annpath=annroot+ann
    picpath=picroot+ann.replace("xml","jpg")
    im = Image.open(picpath)
    img = cv2.imread(picpath)
    draw = ImageDraw.Draw(im)
    DOMTree = xml.dom.minidom.parse(annpath)
    collection = DOMTree.documentElement
    objects = collection.getElementsByTagName("object")
    labelsss = ""
    for object_ in objects:
        #print (object_)
        a=object_.getElementsByTagName("name")[0].childNodes[0].nodeValue
        k=a.split('.',1)
        kk=k[0]        
        b=str(kk) 	
        for i in range(0,len(labelmap)):
            label = labelmap[i]
            print (label)
	    
            if b == label:
	        nn+= 1
		if label not in labelsss:
    		    labelsss+= label +"_"
                bndboxs = object_.getElementsByTagName("bndbox")
                for bndbox in bndboxs:
                    xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
                    ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
                    xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
                    ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
                    xtmp1=xmin.split('.',1)
                    xmin1=xtmp1[0]
                    xtmp2= xmax.split('.', 1)
                    xmax1 = xtmp2[0]
                    xtmp3=ymin.split('.',1)
                    ymin1=xtmp3[0]
                    xtmp4=ymax.split('.',1)
                    ymax1=xtmp4[0]
                xmin = int(xmin1)
                ymin = int(ymin1)
                xmax = int(xmax1)
                ymax = int(ymax1)
                if xmin<0:
                    xmin=0
                if ymin<0:
                    ymin=0
                sp=img.shape
                if xmax>sp[1]:
                    xmax=sp[1]
                if ymax>sp[0]:
                    ymax=sp[0]
 
 
		roiimg=img[ymin: ymax, xmin:xmax]
		save_op = root+'chcc/'+label +"_"+"/"
		mkdir(save_op)
		saveopath = save_op+str(nn)+"_"+ann.replace("xml","jpg")		
		cv2.imwrite(saveopath,roiimg)
 
                draw.rectangle((xmin, ymin, xmax, ymax), outline = colormap[i])
                draw.rectangle((xmin-1, ymin-1, xmax-1, ymax-1), outline = colormap[i])
            	draw.rectangle((xmin+1, ymin+1, xmax+1, ymax+1), outline = colormap[i])
            	draw.rectangle((xmin-2, ymin-2, xmax-2, ymax-2), outline = colormap[i])
            	draw.rectangle((xmin+2, ymin+2, xmax+2, ymax+2), outline = colormap[i])
            	draw.rectangle((xmin-3, ymin-3, xmax-3, ymax-3), outline = colormap[i])
            	draw.rectangle((xmin+3, ymin+3, xmax+3, ymax+3), outline = colormap[i])	
            	break
 
 
        label_has=0
        for label in labelmap:
            if b != label:
                label_has = 1
        if not label_has:
            print (ann+"======"+b+"============================")
       
 
    
    save_p = root+'check/'+labelsss+"/"
    savepath = save_p+ann.replace("xml","jpg")
    mkdir(save_p)
    im.save(savepath)
    #cv2.imwrite(savepath,roiimg)
