import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from colorization_by_learning.NeuralColorizer_NoInput import NeuralColorizer
from colorization_by_learning.colorizers.util import load_img, qt_image_to_array
from colorization_by_custom_method.colorization_by_reference.colorization_by_reference import ColorizationByReference
from colorization_by_custom_method.colorization_by_optimization.colorization_by_optimization import ColorizationByOptimization
from stylesheet import stylesheet
from skimage import color
import cv2
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import time


class Colorizer(QWidget):
    def __init__(self, parent=None):
        super(Colorizer, self).__init__(parent)
        self.setWindowTitle("简单的画板")
        self.painton = True
        self.NeuralColorizer = NeuralColorizer()
        self.OptimizationColorizer = None
        self.ReferenceColorizer = None
        # self.setMouseTracking(True)#默认状态(False):鼠标按下移动mouseMoveEvent才能捕捉到
        self.pos_xy = []
        self.pen = QPen(Qt.black, 2, Qt.SolidLine)
        #
        # self.imageLabel = QLabel()
        # # self.setCentralWidget(self.imageLabel)
        # self.imageLabel.setPixmap(self.image)
        self.originPixmap = None
        self.input = None
        self.reference = None
        self.tool = "Draw"
        self.ChosingMask = False
        self.tmpChosingMask = False
        self.mask_start_pos = (0, 0)
        self.mask_end_pos = (0, 0)
        self.mask_lu = (0, 0)
        self.mask_rd = (0, 0)
        # input/ref mask
        self.ir_mask_start_pos = (0, 0)
        self.ir_mask_end_pos = (0, 0)
        self.input_ref_num = 0
        self.input_input_masks = []
        self.input_ref_masks = []
        self.ir_mask_empty = True

        # self.imageLabel.setMouseTracking(True)
    def colorizeImg(self):
        if(self.radioButtonNN.isChecked()):
            img_obj, img_mask = qt_image_to_array(self.image)
            img_after = self.NeuralColorizer.colorize(self.input, img_obj, img_mask)
            img_after = img_after * 255
            img_after = self.mergeMaskColor(img_after)
            self.lastColorized = img_after
            self.writeNP2Cache(img_after)
            self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))
        elif(self.radioButtonOpt.isChecked()):
            # 优化方法上色
            if not isinstance(self.reference, np.ndarray):
                #交互式图像上色
                t1 = Thread(target=self.optimizationInteractiveColorization, args=())
                t1.start()
            else:
                # 利用事先准备好的图片完成上色
                t1 = Thread(target=self.optimizationPreImageColorization, args=())
                t1.start()
        elif (self.radioButtonOptRecolor.isChecked()):
            print("ReColorize method Opt recolor")
            if not isinstance(self.reference, np.ndarray):
                # 交互式重上色
                t1 = Thread(target=self.optimizationInteractiveReColorization, args=())
                t1.start()
            else:
                # 利用事先准备好的图片完成重上色
                t1 = Thread(target=self.optimizationPreImageReColorization, args=())
                t1.start()
            #交互式图像上色
            #     t1 = Thread(target=self.optimizationInteractiveColorization, args=())
            #     t1.start()

        elif(self.radioButtonOpt_2.isChecked()):
            print("referencecolor")
            self.ReferenceColorizer = ColorizationByReference(self.reference, self.input)
            if len(self.input_ref_masks) == 0:
                # 暂未指定对应参考上色区域，采用全局上色方法
                t1 = Thread(target=self.referenceGlobalColorization, args=())
                t1.start()
            else:
                # 已经指定对应参考上色区域，采用区域对照上色方法
                t1 = Thread(target=self.referenceAreaColorization, args=())
                t1.start()

    def optimizationInteractiveReColorization(self):
        self.OptimizationColorizer = ColorizationByOptimization(self.input)
        img_rgb, img_mask = qt_image_to_array(self.image)
        i = 0
        for each_row in img_mask:
            j = 0
            for each_pixel in each_row:
                if (each_pixel[0] != 0):
                    self.OptimizationColorizer.manual_color(i, j, img_rgb[i][j])
                j = j + 1
            i = i + 1
        recolor_area_info = {'target_row1': self.mask_lu[1], 'target_row2': self.mask_rd[1], 'target_col1': self.mask_lu[0], 'target_col2': self.mask_rd[0]}
        self.OptimizationColorizer.set_recolor_area(recolor_area_info["target_row1"], recolor_area_info["target_row2"],
                                      recolor_area_info["target_col1"], recolor_area_info["target_col2"])
        result_bgr = self.OptimizationColorizer.colorization(is_recolor=True)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = (np.array(result_rgb) * 255).astype(np.uint8)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def optimizationPreImageReColorization(self):
        self.OptimizationColorizer = ColorizationByOptimization(self.input)
        self.OptimizationColorizer.manually_set_marked_image(self.reference)
        result_bgr = self.OptimizationColorizer.colorization(is_recolor=True)
        print(result_bgr)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = (np.array(result_rgb) * 255).astype(np.uint8)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def optimizationPreImageColorization(self):
        print("Colorize method PreImage Optimization")
        self.OptimizationColorizer = ColorizationByOptimization(self.input)
        self.OptimizationColorizer.manually_set_marked_image(self.reference)
        result_bgr = self.OptimizationColorizer.colorization(is_recolor=False)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = (np.array(result_rgb) * 255).astype(np.uint8)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def optimizationInteractiveColorization(self):
        print("Colorize method Optimization")
        self.OptimizationColorizer = ColorizationByOptimization(self.input)
        img_rgb,img_mask = qt_image_to_array(self.image)
        i = 0
        for each_row in img_mask:
            j = 0
            for each_pixel in each_row:
                if(each_pixel[0]!=0):
                    self.OptimizationColorizer.manual_color(i,j,img_rgb[i][j])
                j = j + 1
            i = i + 1

        result_bgr = self.OptimizationColorizer.colorization(is_recolor=False)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = (np.array(result_rgb) * 255).astype(np.uint8)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def referenceGlobalColorization(self):
        result_bgr = self.ReferenceColorizer.global_colorization()
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def referenceAreaColorization(self):
        if self.tool == "RefMask":
            if not (self.ir_mask_start_pos == (0,0) and self.ir_mask_end_pos == (0, 0)):
                self.input_ref_masks.append((self.ir_mask_lu, self.ir_mask_rd))
                self.ir_mask_start_pos = (0, 0)
                self.ir_mask_end_pos = (0, 0)
        elif self.tool == "InputMask":
            if not (self.ir_mask_start_pos == (0, 0) and self.ir_mask_end_pos == (0, 0)):
                self.input_input_masks.append((self.ir_mask_lu, self.ir_mask_rd))
                self.ir_mask_start_pos = (0, 0)
                self.ir_mask_end_pos = (0, 0)
        interactive_area_info = []
        print(self.input_input_masks)
        for i in range(len(self.input_input_masks)):
            each_input_mask = self.input_input_masks[i]
            each_ref_mask = self.input_ref_masks[i]
            interactive_area_info.append({
                'source_row1': each_ref_mask[0][1],
                'source_col1': each_ref_mask[0][0],
                'source_row2': each_ref_mask[1][1],
                'source_col2': each_ref_mask[1][0],
                'target_row1': each_input_mask[0][1],
                'target_row2': each_input_mask[1][1],
                'target_col1': each_input_mask[0][0],
                'target_col2': each_input_mask[1][0]
            })
        print(interactive_area_info)
        result_bgr = self.ReferenceColorizer.area_interactive_colorization(interactive_area_info)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        img_after = self.mergeMaskColor(result_rgb)
        self.lastColorized = img_after
        self.writeNP2Cache(img_after)
        self.img_after.setPixmap(QPixmap("cache.jpg").scaled(self.size[0], self.size[1]))

    def paintEvent(self, event):
        if(self.originPixmap is None):
            return
        if (self.tool == "Draw"):
            painter = QPainter(self.image)
            # painter.begin(self)
            painter.setPen(self.pen)
            if len(self.pos_xy) > 1:
                point_start = self.pos_xy[0]
                for pos_tmp in self.pos_xy:
                    point_end = pos_tmp
                    if point_end == (-1, -1):
                        point_start = (-1, -1)
                        continue
                    if point_start == (-1, -1):
                        point_start = point_end
                        continue
                    painter.drawLine(self.scale * (point_start[0] - 410), self.scale * (point_start[1] - 30),
                                     self.scale * (point_end[0] - 410), self.scale * (point_end[1] - 30))
                    point_start = point_end
        elif (self.tool == "Mask"):
            self.maskmap.fill(Qt.transparent)  # 填充透明色
            start_point = (self.scale * (self.mask_start_pos[0] - 410), self.scale * (self.mask_start_pos[1] - 30))
            start_point = (max(0, min(self.image_size[0], start_point[0])), max(0, min(self.image_size[1], start_point[1])))
            end_point = (self.scale * (self.mask_end_pos[0] - 410), self.scale * (self.mask_end_pos[1] - 30))
            end_point = (max(0, min(self.image_size[0], end_point[0])), max(0, min(self.image_size[1], end_point[1])))
            self.mask_lu = (int(start_point[0]), int(start_point[1]))
            self.mask_rd = (int(end_point[0]), int(end_point[1]))
            left_up = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
            rect = QRect(left_up[0], left_up[1], abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))
            painter = QPainter(self.maskmap)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect)

        elif (self.tool == "RefMask"):
            self.ref_image.fill(Qt.transparent)  # 填充透明色
            painter = QPainter(self.ref_image)
            painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            # draw previous rects
            for start_point, end_point in self.input_ref_masks:
                rect = QRect(start_point[0], start_point[1], abs(end_point[0] - start_point[0]),
                             abs(end_point[1] - start_point[1]))
                painter.drawRect(rect)

            start_point = (self.ref_scale * (self.ir_mask_start_pos[0] - 50), self.ref_scale * (self.ir_mask_start_pos[1] - 250))
            start_point = (max(0, min(self.ref_image_size[0], start_point[0])), max(0, min(self.ref_image_size[1], start_point[1])))
            end_point = (self.ref_scale * (self.ir_mask_end_pos[0] - 50), self.ref_scale * (self.ir_mask_end_pos[1] - 250))
            end_point = (max(0, min(self.ref_image_size[0], end_point[0])), max(0, min(self.ref_image_size[1], end_point[1])))
            self.ir_mask_lu = (int(start_point[0]), int(start_point[1]))
            self.ir_mask_rd = (int(end_point[0]), int(end_point[1]))
            left_up = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
            rect = QRect(left_up[0], left_up[1], abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))
            painter.drawRect(rect)
            # print(start_point, end_point)

            self.ref_pixmapPainter = QPainter(self.ref_accumPixmap)
            self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1], self.ref_originPixmap)
            self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1],
                                              self.ref_image.scaled(self.ref_size[0], self.ref_size[1]))
            self.ref_pixmapPainter.end()
            self.RefereLabel.setPixmap(self.ref_accumPixmap)

        elif (self.tool == "InputMask"):
            self.ir_maskmap.fill(Qt.transparent)  # 填充透明色
            painter = QPainter(self.ir_maskmap)
            painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            # draw previous rects
            for start_point, end_point in self.input_input_masks:
                rect = QRect(start_point[0], start_point[1], abs(end_point[0] - start_point[0]),
                             abs(end_point[1] - start_point[1]))
                painter.drawRect(rect)

            start_point = (self.scale * (self.ir_mask_start_pos[0] - 410), self.scale * (self.ir_mask_start_pos[1] - 30))
            start_point = (max(0, min(self.image_size[0], start_point[0])), max(0, min(self.image_size[1], start_point[1])))
            end_point = (self.scale * (self.ir_mask_end_pos[0] - 410), self.scale * (self.ir_mask_end_pos[1] - 30))
            end_point = (max(0, min(self.image_size[0], end_point[0])), max(0, min(self.image_size[1], end_point[1])))
            self.ir_mask_lu = (int(start_point[0]), int(start_point[1]))
            self.ir_mask_rd = (int(end_point[0]), int(end_point[1]))
            left_up = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]))
            rect = QRect(left_up[0], left_up[1], abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))
            painter.drawRect(rect)
            # print(start_point, end_point)

        # Draw image with guidance
        self.pixmapPainter = QPainter(self.accumPixmap)
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.originPixmap)
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.image.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.maskmap.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.ir_maskmap.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.end()
        self.img_before.setPixmap(self.accumPixmap)

    def mouseMoveEvent(self, event):
        if (self.tool == "Draw"):
            if self.painton:
                pos_tmp = (event.pos().x(), event.pos().y())
                self.pos_xy.append(pos_tmp)
                self.update()
        elif (self.tool == "Mask"):
            if(self.ChosingMask == False):
                self.mask_start_pos = (event.pos().x(), event.pos().y())
                self.mask_end_pos = (event.pos().x(), event.pos().y())
                self.ChosingMask = True
            else:
                self.mask_end_pos = (event.pos().x(), event.pos().y())
        elif (self.tool == "RefMask"):
            if(self.tmpChosingMask == False):
                self.ir_mask_start_pos = (event.pos().x(), event.pos().y())
                self.ir_mask_end_pos = (event.pos().x(), event.pos().y())
                self.tmpChosingMask = True
            else:
                self.ir_mask_end_pos = (event.pos().x(), event.pos().y())
        elif (self.tool == "InputMask"):
            if(self.tmpChosingMask == False):
                self.ir_mask_start_pos = (event.pos().x(), event.pos().y())
                self.ir_mask_end_pos = (event.pos().x(), event.pos().y())
                self.tmpChosingMask = True
            else:
                self.ir_mask_end_pos = (event.pos().x(), event.pos().y())

    def mouseReleaseEvent(self, event):
        if (self.tool == "Draw"):
            pos_test = (-1, -1)  # 用（-1，-1）作为一笔的断点
            self.pos_xy.append(pos_test)
            self.pos_xy = []
            self.update()
        elif (self.tool == "Mask"):
            self.ChosingMask = False
        elif (self.tool == "RefMask"):
            self.tmpChosingMask = False
        elif (self.tool == "InputMask"):
            self.tmpChosingMask = False

    def setupUi(self, Form):
        app.setObjectName("Form")
        app.setStyleSheet(stylesheet)
        self.img_before = QtWidgets.QLabel(Form)
        self.img_before.setGeometry(QtCore.QRect(410, 30, 501, 281))
        self.img_before.setObjectName("img_before")
        self.img_after = QtWidgets.QLabel(Form)
        self.img_after.setGeometry(QtCore.QRect(410, 370, 501, 281))
        self.img_after.setObjectName("img_after")
        self.LoadButton = QtWidgets.QPushButton(Form)
        self.LoadButton.setGeometry(QtCore.QRect(40, 160, 141, 28))
        self.LoadButton.setObjectName("LoadButton")
        self.ProcessButton = QtWidgets.QPushButton(Form)
        self.ProcessButton.setGeometry(QtCore.QRect(210, 490, 141, 28))
        self.ProcessButton.setObjectName("ProcessButton")
        self.ChangeColorButton = QtWidgets.QPushButton(Form)
        self.ChangeColorButton.setGeometry(QtCore.QRect(40, 580, 141, 28))
        self.ChangeColorButton.setObjectName("ChangeColorButton")
        self.ClearColorButton = QtWidgets.QPushButton(Form)
        self.ClearColorButton.setGeometry(QtCore.QRect(40, 610, 141, 28))
        self.ClearColorButton.setObjectName("ClearColorButton")
        self.CameraCaptureButton = QtWidgets.QPushButton(Form)
        self.CameraCaptureButton.setGeometry(QtCore.QRect(40, 190, 141, 28))
        self.CameraCaptureButton.setObjectName("CameraCaptureButton")
        self.LoadVideoButton = QtWidgets.QPushButton(Form)
        self.LoadVideoButton.setGeometry(QtCore.QRect(40, 490, 141, 28))
        self.LoadVideoButton.setObjectName("LoadVideoButton")
        self.CaptureVideoButton = QtWidgets.QPushButton(Form)
        self.CaptureVideoButton.setGeometry(QtCore.QRect(40, 520, 141, 28))
        self.CaptureVideoButton.setObjectName("CaptureVideoButton")
        self.ProcessVideoButton = QtWidgets.QPushButton(Form)
        self.ProcessVideoButton.setGeometry(QtCore.QRect(210, 520, 141, 28))
        self.ProcessVideoButton.setObjectName("ProcessVideoButton")
        self.ClearMaskButton = QtWidgets.QPushButton(Form)
        self.ClearMaskButton.setGeometry(QtCore.QRect(210, 610, 141, 28))
        self.ClearMaskButton.setObjectName("ClearMaskButton")
        self.SelectMaskButton = QtWidgets.QPushButton(Form)
        self.SelectMaskButton.setGeometry(QtCore.QRect(210, 580, 141, 28))
        self.SelectMaskButton.setObjectName("SelectMaskButton")

        self.radioButtonNN = QtWidgets.QRadioButton(Form)
        self.radioButtonNN.setChecked(True)
        self.radioButtonNN.setGeometry(QtCore.QRect(40, 40, 141, 19))
        self.radioButtonNN.setObjectName("radioButtonNN")
        self.LoadReferenceButton = QtWidgets.QPushButton(Form)
        self.LoadReferenceButton.setGeometry(QtCore.QRect(210, 100, 141, 28))
        self.LoadReferenceButton.setObjectName("LoadReferenceButton")
        self.RefereLabel = QtWidgets.QLabel(Form)
        self.RefereLabel.setGeometry(QtCore.QRect(50, 250, 291, 211))
        self.RefereLabel.setObjectName("label")
        self.radioButtonOpt = QtWidgets.QRadioButton(Form)
        self.radioButtonOpt.setGeometry(QtCore.QRect(40, 70, 141, 19))
        self.radioButtonOpt.setObjectName("radioButtonOpt")
        self.radioButtonOpt_2 = QtWidgets.QRadioButton(Form)
        self.radioButtonOpt_2.setGeometry(QtCore.QRect(40, 130, 141, 19))
        self.radioButtonOpt_2.setObjectName("radioButtonOpt_2")
        self.ReferenceMaskButton = QtWidgets.QPushButton(Form)
        self.ReferenceMaskButton.setGeometry(QtCore.QRect(210, 130, 141, 28))
        self.ReferenceMaskButton.setObjectName("ReferenceMaskButton")
        self.InputMaskButton = QtWidgets.QPushButton(Form)
        self.InputMaskButton.setGeometry(QtCore.QRect(210, 160, 141, 28))
        self.InputMaskButton.setObjectName("InputMaskButton")
        self.ClearReferenceMask = QtWidgets.QPushButton(Form)
        self.ClearReferenceMask.setGeometry(QtCore.QRect(210, 190, 141, 28))
        self.ClearReferenceMask.setObjectName("ClearReferenceMask")
        self.title = QtWidgets.QLabel(Form)
        self.title.setGeometry(QtCore.QRect(210, 40, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Condensed")
        font.setPointSize(23)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.radioButtonOptRecolor = QtWidgets.QRadioButton(Form)
        self.radioButtonOptRecolor.setGeometry(QtCore.QRect(40, 100, 141, 19))
        self.radioButtonOptRecolor.setObjectName("radioButtonOptRecolor")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        # Extra
        self.LoadButton.clicked.connect(self.loadFile)
        self.ProcessButton.clicked.connect(self.colorizeImg)
        self.ChangeColorButton.clicked.connect(self.resetColor)
        self.CameraCaptureButton.clicked.connect(self.cameraCapture)
        self.ClearColorButton.clicked.connect(self.clearColor)
        self.LoadVideoButton.clicked.connect(self.loadVideo)
        self.CaptureVideoButton.clicked.connect(self.videoCapture)
        self.ProcessVideoButton.clicked.connect(self.processVideo)
        self.LoadReferenceButton.clicked.connect(self.loadReference)
        self.SelectMaskButton.clicked.connect(self.setToolMask)
        self.ClearMaskButton.clicked.connect(self.clearMask)
        self.InputMaskButton.clicked.connect(self.setToolInputMask)
        self.ReferenceMaskButton.clicked.connect(self.setToolRefMask)
        self.ClearReferenceMask.clicked.connect(self.clearRIMask)
        self.radioButtonNN.clicked.connect(lambda: self.changeApproach(self.radioButtonNN))
        self.radioButtonOpt.clicked.connect(lambda: self.changeApproach(self.radioButtonOpt))
        self.radioButtonOpt_2.clicked.connect(lambda: self.changeApproach(self.radioButtonOpt_2))
        self.label_width = self.img_after.width()
        self.label_height = self.img_after.height()
        self.ref_label_width = self.RefereLabel.width()
        self.ref_label_height = self.RefereLabel.height()

    def changeApproach(self, btn):
        self.radioButtonNN.setChecked(False)
        self.radioButtonOpt.setChecked(False)
        self.radioButtonOpt_2.setChecked(False)
        btn.setChecked(True)

    def clearMask(self):
        self.mask_start_pos = (0, 0)
        self.mask_end_pos = (0, 0)
        self.mask_lu = (0, 0)
        self.mask_rd = (int(self.image_size[0]), int(self.image_size[1]))

    def clearRIMask(self):
        self.input_input_masks = []
        self.input_ref_masks = []
        self.ir_mask_start_pos = (0, 0)
        self.ir_mask_end_pos = (0, 0)
        self.ir_mask_lu = (0, 0)
        self.ir_mask_rd = (0, 0)
        self.ir_maskmap.fill(Qt.transparent)  # 填充透明色
        self.ref_image.fill(Qt.transparent)  # 填充透明色
        self.ref_pixmapPainter = QPainter(self.ref_accumPixmap)
        self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1], self.ref_originPixmap)
        self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1],
                                          self.ref_image.scaled(self.ref_size[0], self.ref_size[1]))
        self.ref_pixmapPainter.end()
        self.RefereLabel.setPixmap(self.ref_accumPixmap)
        self.ir_mask_empty = True

    def setToolMask(self):
        self.tool = "Mask"

    def setToolInputMask(self):
        self.tool = "InputMask"
        if(self.ir_mask_empty == True):
            self.ir_mask_empty = False
        else:
            if self.ir_mask_start_pos == (0,0) and self.ir_mask_end_pos == (0, 0):
                return
            self.input_ref_masks.append((self.ir_mask_lu, self.ir_mask_rd))
            self.ir_mask_start_pos = (0, 0)
            self.ir_mask_end_pos = (0, 0)

    def setToolRefMask(self):
        self.tool = "RefMask"
        if(self.ir_mask_empty == True):
            self.ir_mask_empty = False
        else:
            if self.ir_mask_start_pos == (0,0) and self.ir_mask_end_pos == (0, 0):
                return
            self.input_input_masks.append((self.ir_mask_lu, self.ir_mask_rd))
            self.ir_mask_start_pos = (0, 0)
            self.ir_mask_end_pos = (0, 0)
        #   {
        #       'source_row1': 62, 'source_row2': 90,
        #       'source_col1': 138, 'source_col2': 178,
        #       'target_row1': 111, 'target_row2': 144,
        #       'target_col1': 47, 'target_col2': 101
        #   }

    def setToolDraw(self):
        self.tool = "Draw"

    def resetColor(self):
        self.setToolDraw()
        self.color = QColorDialog.getColor()
        self.pen = QPen(self.color, 2, Qt.SolidLine)

    def clearColor(self):
        self.image.fill(Qt.transparent)  # 填充透明色

    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', './images', 'Image files(*.jpg *.gif *.png *.bmp)')
        if(fname == ""):
            return
        self.input = load_img(fname)
        self.setOriginImage()

    def loadReference(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', './images', 'Image files(*.jpg *.gif *.png *.bmp)')
        if(fname == ""):
            return
        self.reference = load_img(fname)
        self.setReferenceImagge()

    def loadVideo(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', 'c:\\', 'Image files(*.avi *.gif *.mp4)')
        if(fname == ""):
            return
        self.loadVideoFromFile(fname)

    def loadVideoFromFile(self, fname):
        self.cap = cv2.VideoCapture(fname)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.input = image
                self.setOriginImage()

    def processVideo(self):
        t1 = Thread(target=self.processVideoThread, args=())
        t1.start()

    def processVideoThread(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.input = image
                self.setOriginImage()
                self.colorizeImg()
                cv2.waitKey(1)
            else:
                break
        self.cap.release()
        cv2.destroyWindow('frame')

    def setReferenceImagge(self):
        height = self.reference.shape[0]
        width = self.reference.shape[1]
        scale = self.ref_label_height / height
        if self.ref_label_width / width < self.ref_label_height / height:
            scale = self.ref_label_width / width
        self.ref_scale = 1 / scale
        self.ref_size = (int(scale * width), int(scale * height))
        self.writeNP2Cache(self.reference)
        self.ref_originPixmap = QPixmap("cache.jpg").scaled(self.ref_size[0], self.ref_size[1])
        self.ref_image = QPixmap(width, height)
        self.ref_image_size = (width, height)
        self.ref_image.fill(Qt.transparent)  # 填充透明色
        self.ref_accumPixmap = QPixmap(self.ref_size[0], self.ref_size[1])
        self.ref_pixmapPainter = QPainter(self.ref_accumPixmap)
        self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1], self.ref_originPixmap)
        self.ref_pixmapPainter.drawPixmap(0, 0, self.ref_size[0], self.ref_size[1], self.ref_image.scaled(self.ref_size[0], self.ref_size[1]))
        self.ref_pixmapPainter.end()
        self.RefereLabel.setPixmap(self.ref_accumPixmap)

    def setOriginImage(self):
        height = self.input.shape[0]
        width = self.input.shape[1]
        scale = self.label_height / height
        if self.label_width / width < self.label_height / height:
            scale = self.label_width / width
        self.scale = 1 / scale
        self.size = (int(scale * width), int(scale * height))
        self.lastColorized = self.input
        self.writeNP2Cache(self.input)
        self.originPixmap = QPixmap("cache.jpg").scaled(self.size[0], self.size[1])
        self.image = QPixmap(width, height)
        self.image.fill(Qt.transparent)  # 填充透明色
        self.image_size = (width, height)
        self.maskmap = QPixmap(width, height)
        self.maskmap.fill(Qt.transparent)  # 填充透明色
        self.ir_maskmap = QPixmap(width, height)
        self.ir_maskmap.fill(Qt.transparent)  # 填充透明色
        self.accumPixmap = QPixmap(self.size[0], self.size[1])
        self.pixmapPainter = QPainter(self.accumPixmap)
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.originPixmap)
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.image.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.maskmap.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.drawPixmap(0, 0, self.size[0], self.size[1], self.ir_maskmap.scaled(self.size[0], self.size[1]))
        self.pixmapPainter.end()
        self.img_before.setPixmap(self.accumPixmap)
        self.mask_lu = (0, 0)
        self.mask_rd = (int(self.image_size[0]), int(self.image_size[1]))

    def videoCapture(self):
        t1 = Thread(target=self.videoCaptureTread, args=())
        t1.start()

    def videoCaptureTread(self):
        cap = cv2.VideoCapture(0)
        # Before start recording
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('record', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        # Input
        fps = 30
        height = frame.shape[0]
        width = frame.shape[1]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('recorded.mp4', fourcc, fps, (width, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('record', frame)
                out.write(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release
        cap.release()
        out.release()
        cv2.destroyWindow('record')
        self.loadVideoFromFile('recorded.mp4')

    def cameraCapture(self):
        t1 = Thread(target=self.cameraCaptureThread, args=())
        t1.start()

    def cameraCaptureThread(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 900)
        cap.set(4, 900)
        while (cap.isOpened()):
            ret_flag, Vshow = cap.read()
            cv2.imshow('Capture', Vshow)
            k = cv2.waitKey(1)
            if k == ord('q'):
                image = cv2.cvtColor(Vshow, cv2.COLOR_RGB2BGR)
                self.input = image
                break
        cap.release()
        cv2.destroyWindow('Capture')
        self.setOriginImage()

    def mergeMaskColor(self, img_after):
        img = np.array(self.lastColorized)
        img.flags.writeable = True
        img[self.mask_lu[1]:self.mask_rd[1], self.mask_lu[0]:self.mask_rd[0], :] \
            = img_after[self.mask_lu[1]:self.mask_rd[1], self.mask_lu[0]:self.mask_rd[0], :]
        return img

    def writeNP2Cache(self, image):
        img = image.astype("uint8")
        im = Image.fromarray(img)
        im.save("cache.jpg")

    def load_text(self):
        print("load--text")
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter(QDir.Files)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            with f:
                data = f.read()
                self.content.setText(data)

    def createVideoWriter(self, name):
        fps = 30
        height = self.image_size[1]
        width = self.image_size[0]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == True:
        #         cv2.imshow('frame', frame)
        #         out.write(frame)
        #         if cv2.waitKey(10) & 0xFF == ord('q'):
        #             break
        #     else:
        #         break

    def releaseVideoWriter(self):
        self.out.release()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.img_before.setText(_translate("Form", "Input Image"))
        self.img_after.setText(_translate("Form", "Colorized Image"))
        self.LoadButton.setText(_translate("Form", "Load Image"))
        self.ProcessButton.setText(_translate("Form", "Colorize Image"))
        self.ChangeColorButton.setText(_translate("Form", "Change Color"))
        self.ClearColorButton.setText(_translate("Form", "Clear Color"))
        self.CameraCaptureButton.setText(_translate("Form", "Capture Image"))
        self.LoadVideoButton.setText(_translate("Form", "Load Video"))
        self.CaptureVideoButton.setText(_translate("Form", "Capture Video"))
        self.ProcessVideoButton.setText(_translate("Form", "Colorize Video"))
        self.ClearMaskButton.setText(_translate("Form", "Clear Mask"))
        self.SelectMaskButton.setText(_translate("Form", "Select Mask"))
        self.radioButtonNN.setText(_translate("Form", "Neural Approach"))
        self.LoadReferenceButton.setText(_translate("Form", "Load Reference"))
        self.RefereLabel.setText(_translate("Form", "Reference Image"))
        self.radioButtonOpt.setText(_translate("Form", "Optimize Approach"))
        self.radioButtonOpt_2.setText(_translate("Form", "Reference"))
        self.ReferenceMaskButton.setText(_translate("Form", "Reference Mask"))
        self.InputMaskButton.setText(_translate("Form", "Input Mask"))
        self.ClearReferenceMask.setText(_translate("Form", "Clear RefMask"))
        self.title.setText(_translate("Form", "COLORIZER"))
        self.radioButtonOptRecolor.setText(_translate("Form", "Opt Recolor"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # Form = Colorizer()
    ui = Colorizer()
    Form = ui
    # Form = QtWidgets.QWidget()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
