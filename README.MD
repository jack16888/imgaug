1, 做VOC数据集训练时候的总结

2, 做数据增强时候的总结，jpgdata_aug.py

 changefilename.py

 changexmljpgname.py

 changfilename.py

 color.py image color aug

 image_aug.py imge aug

 jpgdata_aug.py use imgaug package to do image data augmentation ,change xml and jpa , for voc dataset

 makemain.py build train val test train val in Imagesets/Main/

 python.py

 README.MD

 xml_aug.py script for draw xml piont to pictures.


  
4, color_brightness,py 这个文件操作后，会把亮度，对比度，sharpness 这些做增强，同时把对应的标签文件和jpg文件进行

相应的保存，但是这个时候save的jpg文件和xml里的xml文件只是名字匹配上了，xml内部的文件名还需要修改

这时候，采用  changexmljpgname.py 这个文件，通过xml文件名，修改内部的name。

color_brightness,py changexmljpgname.py changefilename.py 这三个文件按顺序使用。

5 , jpgdata_aug.py 这个文件，只要执行了，就可以直接生成旋转，亮度之类的图像增强，对应的图片和xml文件都是匹配好了的。

6, 仔细分析jpgdata_aug2.py 这个文件，放大1.5倍数是因为ssd224×224的数据集大多为500×300的，要适应到mobilenetssd448×448的情况

  ,应该可以满足各种增强需求。SSD的不用考虑其他方法了。https://blog.csdn.net/u010451780/article/details/106249758

3, 参考：
https://github.com/aleju/imgaug

pip3 install git+https://github.com/aleju/imgaug

https://github.com/aleju/imgaug

https://blog.csdn.net/limiyudianzi/article/details/86497305

https://mp.csdn.net/console/editor/html/106214112

https://github.com/cyndi088/DataAugmentation/blob/master/augmentation.py

https://blog.csdn.net/u010451780/article/details/106249758
