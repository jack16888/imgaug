import os
import shutil
 
## datadir AND savedir CAN NOT BE SAME
img_datadir="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/JPEGImages/"
img_savedir="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/newJPEGImages/"
 
xml_datadir="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/Annotations/"
xml_savedir="/home/zhangwanchun/data/VOCdevkit/VOC2007_aug/newAnnotations/"
 
imglist=os.listdir(img_datadir)
k=0
for img in imglist:
 name=img
 k=k+1
 print k
# zzname=img.split("_",2)
 img_datapath=img_datadir+name
 xml_datapath=xml_datadir+name.replace("jpg","xml")
# img_save_name='mhq_171215_'+str(k)+"_"+zzname[2]
# xml_save_name=img_save_name.replace(".jpg",".xml")
 
 img_save_name='888_'+str(k)+".jpg"
 xml_save_name=img_save_name.replace(".jpg",".xml")
 
 img_savepath=img_savedir+img_save_name 
 xml_savepath=xml_savedir+xml_save_name 
 
 #if os.path.exists(img_datapath):
 shutil.copy(img_datapath, img_savepath)
  
 #if os.path.exists(xml_datapath):
 shutil.copy(xml_datapath, xml_savepath)  
