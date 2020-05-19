# coding: utf-8
import sys
import os  
import os.path  
import xml.dom.minidom  
from xml.dom.minidom import parse
import xml.dom.minidom
import os,shutil
#import numpy as np
#import cv2
import urllib
from PIL import Image, ImageDraw
 
path="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/newAnnotations"  
reload(sys)                         # 2
sys.setdefaultencoding('utf-8')     # 3
files=os.listdir(path)
s=[]
num=0
for xmlFile in files:
    num+=1
    print(num)
    imgname=xmlFile.replace(".xml",".jpg")
    if not os.path.isdir(xmlFile): 
        print (xmlFile)
        dom=xml.dom.minidom.parse(os.path.join(path,xmlFile))
        root=dom.documentElement  
        #filename = root.getElementsByTagName("filename").childNodes[0].nodeValue
        #print (filename)
        filename1=root.getElementsByTagName('filename')
        n0=filename1[0]
        print (n0.firstChild.data)
 
	
        a=imgname
        n0.firstChild.data=a
        print (n0.firstChild.data)
    
 
        with open(os.path.join(path, xmlFile), 'w') as fh:
            dom.writexml(fh)
