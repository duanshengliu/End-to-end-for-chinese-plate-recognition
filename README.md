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
