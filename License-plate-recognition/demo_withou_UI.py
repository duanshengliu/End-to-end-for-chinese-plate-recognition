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
from PIL import Image, ImageDraw, ImageFont
unet = keras.models.load_model('E:/python3/TensorFlow1/end2end-alpr-ch/License-plate-recognition/unet.h5')
cnn = keras.models.load_model('E:/python3/TensorFlow1/end2end-alpr-ch/License-plate-recognition/cnn.h5')
#C:/Users/dell/Desktop/complex
#E:\python3\TensorFlow1\end2end-alpr-ch\CCPD2019\CCPD2019\ccpd_base
input_dir = 'C:/Users/dell/Desktop/complex/w'
output_dir = 'C:/Users/dell/Desktop/complex/w/res'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

index = 1
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
            x = int(gtbox[4]*(512/720))
            y = int(gtbox[5]*(512/1160))
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
            Lic_pred = cnn_predict(cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
            #print('print the result')
            #print(Lic_pred)
            if Lic_pred:
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
                    elif i == 1:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
                    elif i == 2:
                        text=lic_pred[1]
                        text = text[0:2]+text[3:]
            else:  # Lic_pred为空说明未能识别
                text='Cannot be Detected'
                #print(text)
            cv2img = cv2.cvtColor(img_src_copy, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
            pilimg = Image.fromarray(cv2img)
 
            # PIL图片上打印汉字
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
            draw.text((x, y-25), text, (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
 
            # PIL图片转cv2 图片
            cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

            # cv2.putText(img_src_copy, text, (x, y-10), cv2.FONT_ITALIC, 1, (0, 255, 0))

            # img_src, img_mask = unet_predict(unet, img_src_path)
            # img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
            # print(img_mask)
            # cv2.imshow('image1',img_mask)
            # cv2.imwrite('C:/Users/dell/Desktop/img_msk.jpg',img_mask)
            # cv2.imshow('image2',img_src)
            # cv2.imshow('image3',img_src_copy)
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', cv2charimg)
            index += 1
            # print(Lic_img)
            # cv2.waitKey(0)
            # key = cv2.waitKey(30) & 0xff
            # if key == 27:
            #    sys.exit(0)
            #if index == maxn:
            #    break