# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BscanObjDetection.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
import cv2

from ultralytics import YOLO


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1131, 932)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 20, 98, 26))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(390, 20, 98, 26))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 180, 511, 451))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(590, 180, 491, 461))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1131, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 点击响应函数
        self.pushButton.clicked.connect(self.uploadImage)
        self.pushButton_3.clicked.connect(self.startProgram)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Upload"))
        self.pushButton_3.setText(_translate("MainWindow", "Detect"))

    def uploadImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'select images', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        if image_path:
            # 在这里添加加载图片的逻辑，例如显示图片到label
            pixmap = QtGui.QPixmap(image_path)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)


    def startProgram(self):
        model = YOLO('runs/train/exp/weights/best.pt')
        results = model(self.image_path)
        annotated_frame = results[0].plot()
        # 图片暂存
        save_image_name = self.image_path.split("/")[-1].split(".")[0] + "_pred"
        cv2.imwrite(f"predict_result\{save_image_name}.jpg", annotated_frame)

        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        pixmap = QtGui.QPixmap.fromImage(qimage)

        # 都执行：
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())