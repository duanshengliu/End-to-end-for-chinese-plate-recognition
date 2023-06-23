'''
python:3.7.6
'''

import cv2


def main():
    # 创建VideoCapture对象，参数为摄像头索引号，0表示第一个摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        # 如果视频帧读取成功
        if ret:
            # 显示视频帧
            cv2.imshow('Camera', frame)

        # 按下Esc键退出循环
        if cv2.waitKey(1) == 27:
            break

    # 释放VideoCapture对象和销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import cv2
from PIL import Image, ImageTk


def display(img_src_path, img_src_copy, Lic_img, Lic_pred, canvas_src, canvas_lic1, canvas_lic2, canvas_lic3,
            canvas_pred1, canvas_pred2, canvas_pred3):
    if img_src_path is None:  # 还没选择图片就进行预测
        canvas_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
    else:
        if Lic_pred:
            img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
            img_Tk = ImageTk.PhotoImage(img)
            canvas_src.delete('all')  # 显示前,先清空画板
            canvas_src.create_image(258, 258, image=img_Tk, anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
            for i, lic_pred in enumerate(Lic_pred):
                if i == 0:
                    lic_Tk1 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                    canvas_lic1.create_image(5, 5, image=lic_Tk1, anchor='nw')
                    canvas_pred1.create_text(35, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                elif i == 1:
                    lic_Tk2 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                    canvas_lic2.create_image(5, 5, image=lic_Tk2, anchor='nw')
                    canvas_pred2.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                elif i == 2:
                    lic_Tk3 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                    canvas_lic3.create_image(5, 5, image=lic_Tk3, anchor='nw')
                    canvas_pred3.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
        else:  # Lic_pred为空说明未能识别
            canvas_pred1.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))
