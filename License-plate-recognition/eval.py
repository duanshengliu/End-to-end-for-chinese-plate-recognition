import cv2
import os
import sys
import numpy as np
import numpy as np
from shapely.geometry import Polygon,MultiPoint  #多边形
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow import keras
from core import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict
from time import time
unet = keras.models.load_model('E:/python3/TensorFlow1/end2end-alpr-ch/License-plate-recognition/unet.h5')
cnn = keras.models.load_model('E:/python3/TensorFlow1/end2end-alpr-ch/License-plate-recognition/cnn.h5')
#C:/Users/dell/Desktop/complex
#E:\python3\TensorFlow1\end2end-alpr-ch\CCPD2019\CCPD2019\ccpd_base
input_dir = 'C:/Users/dell/Desktop/complex/not'
# output_dir = 'C:/Users/dell/Desktop/result'
# if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

def get_iou(a,b):
	'''
	:param a: box a [x0,y0,x1,y1,x2,y2,x3,y3]
	:param b: box b [x0,y0,x1,y1,x2,y2,x3,y3]
	:return: iou of bbox a and bbox b
	'''
	a = a.reshape(4, 2)
	poly1 = Polygon(a).convex_hull
	
	b = b.reshape(4, 2)
	poly2 = Polygon(b).convex_hull
	
	if not poly1.intersects(poly2):  # 如果两四边形不相交
		iou = 0
	else:
		try:
			inter_area = poly1.intersection(poly2).area  # 相交面积
			union_area = poly1.area + poly2.area - inter_area
			if union_area == 0:
				iou = 0
			else:
				iou = inter_area / union_area
		except shapely.geos.TopologicalError:
			print('shapely.geos.TopologicalError occured, iou set to 0')
			iou = 0
	return iou

index = 1
right = 0
nodet = 0
false = 0
iou_right = 0
# maxn = 100001
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
start = time()
for(path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processing picture %s' %index)
            img_src_path = path+'/'+filename
            gtpoly = filename.split('/')[-1].rsplit('.', 1)[0].split('-')[-4]
            gtpoly = gtpoly.split('_')
            gtbox = []
            for i in range(4):
                gtbox.extend(gtpoly[i].split('&'))
            gtbox = list(map(int,gtbox))
            gtbox = np.array(gtbox)
            # print(gtpoly)
            # print(gtbox)
            gt = filename.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
            gt=gt.split('_')
            gt = list(map(int,gt))
            lpn = provinces[gt[0]] + alphabets[gt[1]] + ads[gt[2]] + ads[gt[3]] + ads[gt[4]] + ads[gt[5]] + ads[gt[6]]
            # img = cv2.imread(img_src_path)
            # img = cv2.resize(img,(512,512))
            # cv2.imwrite(img_src_path,img)
            img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
                img_src_copy, Lic_img = img_src, [lic]
            else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                img_src, img_mask = unet_predict(unet, img_src_path)
                img_src_copy, Lic_img, prebox = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
            prebox = np.array(prebox)
            if get_iou(prebox,gtbox) > 0.7:
                iou_right += 1
            Lic_pred = cnn_predict(cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
            #print('print the result')
            #print(Lic_pred)
            if Lic_pred:
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
                        #print(text)
                    elif i == 1:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
                        #print(text)
                    elif i == 2:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
                        #print(text)
            else:  # Lic_pred为空说明未能识别
                text='未能识别'
                #print(text)
                nodet += 1
            if text == lpn:
                right += 1
            else:
                false += 1
            # img_src, img_mask = unet_predict(unet, img_src_path)
            # img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
            # print(img_mask)
            # cv2.imshow('image1',img_mask)
            # cv2.imwrite('C:/Users/dell/Desktop/img_msk.jpg',img_mask)
            # cv2.imshow('image2',img_src)
            # cv2.imshow('image3',img_src_copy)
            # cv2.imwrite(output_dir+'/'+str(index)+'.jpg', img_src_copy)
            index += 1
            # print(Lic_img)
            # cv2.waitKey(0)
            # key = cv2.waitKey(30) & 0xff
            # if key == 27:
            #    sys.exit(0)
            #if index == maxn:
            #    break
index = index - 1
t = (time() - start)/index
print('index: ', index)
print('the fps: ', t)
print('right recognitons: ', right)
print('false recognitions: ', false)
print('not detected: ', nodet)
print('right/index: ', right/index)
print('iou_right: ', iou_right)
print('iou_right/index: ', iou_right/index)