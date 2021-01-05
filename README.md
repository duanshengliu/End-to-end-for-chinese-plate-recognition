# End-to-end-for-chinese-plate-recognition

## 基于u-net，cv2以及cnn的中文车牌定位，矫正和端到端识别软件，其中unet和cv2用于车牌定位和矫正，cnn进行车牌识别，unet和cnn都是基于tensorflow的keras实现
## 环境：python:3.6, tensorflow:1.15.2, opencv: 4.1.0.25, keras: 2.3.1
### 整体思路：1. 利用u-net图像分割得到二值化图像，2. 再使用cv2进行边缘检测获得车牌区域坐标，并将车牌图形矫正，3. 利用卷积神经网络cnn进行车牌多标签端到端识别，具体描述可见CSDN博客：https://blog.csdn.net/qq_32194791/article/details/106748685
### 实现效果：拍摄角度倾斜、强曝光或昏暗环境等都能较好地识别，甚至有些百度AI车牌识别未能识别的图片也能识别

### 注意：若是直接识别类似下图的无需定位的完整车牌，那么请确保图片尺寸小于等于240 * 80，否则会被认为图片中含其余区域而进行定位，反而识别效果不佳

![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/lic.png) 
### 其余的没什么问题，正常识别都可以

### 部分效果图：
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/0.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/1.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/2.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/3.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/4.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/5.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/6.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/7.png)

### 模型评估

我使用了来自[CCPD(你可以点击此处查看CCPD介绍)](https://github.com/detectRecog/CCPD)的车牌图片进行模型评估，并得到了最终结果。模型的FPS在我的机器（GTX1050 以及 Intel(R)Core(TM) i7-8750H CPU @ 2.20GHz）上只达到了14，

### 检测结果

使用u-net和cv2的网络模型在ccpd上的评估结果如下表所示，只有当IOU大于等于0.7的时候，在大部分的论文里面是将IOU设置为大于0.5就会被判定为检测正确，但是我使用了更高的门槛，只有达到0.7才会被认为检测正确。同时也可以看到模型的表现在base和weather两个数据集上表现优秀。

<center>表一：在CCPD上的车牌定位评估结果</center>

| 指标\数据集类型 | base  | weather | tilt  | rotate | fn    | db    | challenge |
| --------------- | ----- | ------- | ----- | ------ | ----- | ----- | --------- |
| AP              | 98.69 | 96.44   | 75.81 | 87.97  | 63.35 | 37.67 | 74.51     |

### 识别结果

值得注意的是，这里我的识别结果是直接针对所有的图片，因为在大部分的论文中在评估识别模型的时候，一般会选择在检测阶段IOU大于0.6或者其他数值的车牌图片来进行进一步的识别评估。

同时经过评估也发现，大部分的字符错误都集中在第一个字符错误，如果只考虑后面六位字母数字的字符那么结果可以会更好。

<center>表二：在CCPD上的车牌字符识别评估结果</center>

| 指标\数据集类型 | base  | weather | tilt | rotate | fn    | db    | challenge |
| --------------- | ----- | ------- | ---- | ------ | ----- | ----- | --------- |
| AP              | 92.43 | 85.66   | 44.7 | 64.82  | 30.46 | 13.26 | 24.65     |

