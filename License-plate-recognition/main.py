import sys
import cv2
from myui import *
from core import locate_and_correct
from Unet import unet_predict


class MyLpr(MyUI):
    def __int__(self):
        super().__init__()

    def display(self):
        # 还没选择图片就进行预测
        if self.img_src_path is None:
            self.can_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
            return

        img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
        h, w = img_src.shape[0], img_src.shape[1]

        # 情况一 : 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
        if h * w <= 240 * 80 and 2 <= w / h <= 5:  #
            lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
            img_src_copy, Lic_img = img_src, [lic]

        # 情况二 : 需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
        else:
            img_src, img_mask = unet_predict(self.unet, self.img_src_path)
            # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
            img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)

        Lic_pred = cnn_predict(self.cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
        # 情况一 : 识别失败
        if Lic_pred is None:
            self.can_pred1.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))
            return

        # 情况二 : 识别成功
        img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
        self.img_Tk = ImageTk.PhotoImage(img)
        self.can_src.delete('all')  # 显示前,先清空画板
        self.can_src.create_image(258, 258, image=self.img_Tk,
                                  anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
        for i, lic_pred in enumerate(Lic_pred):
            if i == 0:
                self.lic_Tk1 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                self.can_pred1.create_text(35, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
            elif i == 1:
                self.lic_Tk2 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                self.can_lic2.create_image(5, 5, image=self.lic_Tk2, anchor='nw')
                self.can_pred2.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
            elif i == 2:
                self.lic_Tk3 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                self.can_lic3.create_image(5, 5, image=self.lic_Tk3, anchor='nw')
                self.can_pred3.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))


def close():
    keras.backend.clear_session()
    sys.exit(0)


a = MyLpr()
a.protocol("WM_DELETE_WINDOW", close)
a.mainloop()
