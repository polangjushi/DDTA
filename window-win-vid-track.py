# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QTabWidget, QMessageBox, QFileDialog, QApplication, QMainWindow, QLabel
from threading import Event, Thread
from os.path import relpath, join, isfile
from os import listdir
from sys import path, exit, argv
from pathlib import Path
from cv2 import imread, resize, imwrite
from torch import zeros, from_numpy, tensor
import time
import sys
import os

import matplotlib
matplotlib.use('Agg')  # for writing to files only
import matplotlib.pyplot as plt
import cv2
from torch.backends import cudnn
from queue import Queue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH

if str(ROOT / 'deep_sort') not in sys.path:
    sys.path.append(str(ROOT / 'deep_sort'))  # add deep_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
# yolov5
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
from trackers.ocsort_tracker.ocsort import OCSort

from GUI.mainwin_vid import Ui_MainWindow
from GUI.tools import MyLabel, stop_thread, frameToQImage, MyFigure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
import threading
import time


# 添加一个关于界面
# 窗口主类
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.root = os.getcwd()
        self.imgSavePath = 'results/image'
        self.vidSavePath = 'results/video'
        self.original_size = [1024, 1024]
        self.input_size = [1024, 1024]
        self.output_size = 1024
        self.img_path = ""
        self.vid_path = ""
        self.det_weight = ""
        self.device = "cpu"
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = Event()
        self.webcam = True
        self.stopEvent.clear()
        self.imgCalibration = 0.0
        self.vidCalibration = 0.0
        self.isImg = False
        self.diameters = []
        self.preDiameters = []
        self.vidDiameters = []
        self.aveDiameter = 0.0
        self.arrowDatas = []
        self.model = None
        self.iou = 0.45
        self.conf = 0.25
        self.lineWidth = 2
        self.num = 0
        self.currentNum = 0
        self.paintScope = [0.0, 1.0, 0.0, 1.0]
        self.rawVideo = None
        self.vidDecThread = None
        self.timerVideo = QTimer()
        self.isVideoPalying = False
        self.frame = None
        self.realTimeDetect = False
        self.stopVidDetFlag = False
        self.outputVidFPS = 30

        self.ocsortDetThr = 0.65
        self.ocsortIouThr = 0.2
        self.pauseFlag = False

        self.trackLength = 60
        self.tracks = {}

        self.setupUi(self)
        self.setMyLabel()
        self.setFigure()
        self.initUI()

    '''
    初始化
    '''

    # 初始化自定义 Label
    def setMyLabel(self):
        # 图像label
        self.left_img = MyLabel(self.frame)
        self.left_img.setStyleSheet("")
        self.left_img.setText("")
        self.left_img.setAlignment(Qt.AlignCenter)
        # self.left_img.setScaledContents(True)
        self.left_img.setObjectName("left_img")
        self.imgVerticalLayout.addWidget(self.left_img)
        # 视频label
        self.mainVid = MyLabel(self.tab_2)
        # self.mainVid.setStyleSheet("background:grey")
        self.mainVid.setFrameShadow(QtWidgets.QFrame.Plain)
        self.mainVid.setText("")
        self.mainVid.setAlignment(Qt.AlignCenter)
        # self.mainVid.setScaledContents(True)
        self.mainVid.setObjectName("mainVid")
        self.vidVerticalLayout.addWidget(self.mainVid)

    def setFigure(self):
        self.disFig = MyFigure()
        valToolbar = NavigationToolbar(self.disFig, self)
        self.disFig.axes = self.disFig.fig.add_axes([0.12, 0.1, 0.8, 0.85])
        self.disFig.axes.set_xlabel('drop size(mm)')
        self.disFig.axes.set_ylabel('drop number')
        self.verticalLayout_19.addWidget(self.disFig)
        self.verticalLayout_19.addWidget(valToolbar)

        self.vidDisFig = MyFigure()
        self.vidDisFig.axes = self.vidDisFig.fig.add_axes([0.1, 0.2, 0.85, 0.75])
        self.vidDisFig.axes.set(xlim=[0, 10], ylim=[0, 30],
                                ylabel='drop number', xlabel='drop size(mm)')
        self.verticalLayout_11.addWidget(self.vidDisFig)

        self.arrowFig = MyFigure()
        arrowToolbar = NavigationToolbar(self.arrowFig, self)
        self.arrowFig.axes = self.arrowFig.fig.add_axes([0.1, 0.1, 0.85, 0.85])
        self.arrowFig.axes.set(xlim=[0, 1], ylim=[0, 1])
        self.verticalLayout_24.addWidget(self.arrowFig)
        self.verticalLayout_24.addWidget(arrowToolbar)

    # 响应初始化
    def initUI(self):
        # 检测图片
        self.up_img_button.clicked.connect(self.upload_img)
        self.con_cali_button.clicked.connect(self.setImageScale)
        self.det_img_button.clicked.connect(self.detect_img)
        self.set_cali_button.clicked.connect(self.paintScaleState)
        self.pic_model_load_button.clicked.connect(self.modelInit)
        self.picIouSlider.valueChanged.connect(self.sliderChange)
        self.picConfSlider.valueChanged.connect(self.sliderChange)
        self.picIouSpinBox.valueChanged.connect(self.iouSpinBoxChange)
        self.picConfSpinBox.valueChanged.connect(self.confSpinBoxChange)
        self.picIouSpinBox.setValue(self.iou)
        self.picConfSpinBox.setValue(self.conf)
        self.detectLineWidthSpinBox.valueChanged.connect(self.lineWidthChange)
        self.detectLineWidthSpinBox.setValue(self.lineWidth)
        self.lineEdit_2.setText(str(self.input_size[0]))
        self.lineEdit_3.setText(str(self.input_size[1]))
        self.lineEdit_2.textChanged.connect(self.inputSizeChange)
        self.clearButton.clicked.connect(self.clearAllResults)
        self.getPicSizeButton.clicked.connect(self.getPicSize)
        self.gpuRadioButton.clicked.connect(self.setDevice)
        self.cpuRaidoButton.clicked.connect(self.setDevice)
        self.freeRadioButton.clicked.connect(self.paintStatusChange)
        self.verticalRadioButton.clicked.connect(self.paintStatusChange)
        self.horizontalRadioButton.clicked.connect(self.paintStatusChange)
        self.detWeightsComboBox.activated.connect(self.selectWeights)
        self.searchWeights()
        self.withdrawButton.clicked.connect(self.withdrawResult)
        self.setPlotScopePushButton.clicked.connect(self.paintScopeState)
        self.paintScope = self.mainVid.scopeData
        self.conScopePushButton.clicked.connect(self.setPaintScope)
        self.saveButton.clicked.connect(self.outputResults)
        self.tabWidget.currentChanged.connect(self.changeTabWidgetPage)
        self.statusbar.showMessage("就绪")

        # 检测视频
        self.loadVidPushButton.clicked.connect(self.upload_vid)
        self.detVidPushButton.clicked.connect(self.detVideoThread)
        self.playVidPushButton.clicked.connect(self.startStopVideo)
        self.stopVidDetButton.clicked.connect(self.stopVidDetThread)
        self.vidProgressHorizontalSlider.valueChanged.connect(self.setVideoProgressBar)
        self.saveFramePushButton.clicked.connect(self.savePreFrame)
        self.realTimedetBtn.clicked.connect(self.setRealTimeDetect)
        self.trackConfSpinBox.valueChanged.connect(self.setOcsortDetThr)
        self.trackIouSpinBox.valueChanged.connect(self.setOcsortIouThr)
        self.pauseVidDetButton.clicked.connect(self.setPauseStatus)

    # 模型初始化
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        self.logTextEdit.append("加载模型：" + weights + "  --Device: " + self.device)
        self.vidLogTextEdit.append("加载模型：" + weights + "  --Device: " + self.device)
        return model

    # 模型初始化函数
    def modelInit(self):
        if self.det_weight == "":
            pass
        else:
            self.model = self.model_load(weights="weights/" + self.det_weight,
                                         device=self.device)

    # tabwidget 页面切换响应
    def changeTabWidgetPage(self):
        if self.tabWidget.currentIndex() == 0:
            if self.mainVid.scaleRes == True:
                self.mainVid.scaleRes = False
                self.left_img.scaleRes = True

        if self.tabWidget.currentIndex() == 1:
            if self.left_img.scaleRes == True:
                self.left_img.scaleRes = False
                self.mainVid.scaleRes = True

    # 绘制按钮点击响应
    def paintScaleState(self):
        if self.tabWidget.currentIndex() == 0:
            if self.left_img.scaleRes == True:
                self.left_img.scaleRes = False
                self.set_cali_button.setStyleSheet("")
            else:
                self.left_img.scaleRes = True
                self.set_cali_button.setStyleSheet("QPushButton{background:yellow;}")

        if self.tabWidget.currentIndex() == 1:
            if self.mainVid.scopeRes == True:
                self.mainVid.scopeRes = False
                self.setPlotScopePushButton.setStyleSheet("")

            if self.mainVid.scaleRes == True:
                self.mainVid.scaleRes = False
                self.set_cali_button.setStyleSheet("")
            else:
                self.mainVid.scaleRes = True
                self.set_cali_button.setStyleSheet("QPushButton{background:yellow;}")

    def paintScopeState(self):
        if self.mainVid.scaleRes == True:
            self.mainVid.scaleRes = False
            self.set_cali_button.setStyleSheet("")

        if self.mainVid.scopeRes == True:
            self.setPlotScopePushButton.setStyleSheet("")
            self.mainVid.scopeRes = False

        else:
            self.setPlotScopePushButton.setStyleSheet("QPushButton{background:yellow;}")
            self.mainVid.scopeRes = True

    def caliInfo(self):
        QMessageBox.information(self, "ERROR", "请先校正!", QMessageBox.Yes)

    def noImgInfo(self):
        QMessageBox.information(self, "ERROR", "未加载图片!", QMessageBox.Yes)

    def noVidInfo(self):
        QMessageBox.information(self, "ERROR", "未加载视频!", QMessageBox.Yes)

    def noMoelInfo(self):
        QMessageBox.information(self, "ERROR", "未加载模型!", QMessageBox.Yes)

    # 设置标尺响应

    def setImageScale(self):
        if self.tabWidget.currentIndex() == 0:

            if self.lineEdit.text() != '' and float(self.lineEdit.text()) > 0.0:
                self.imgCalibration = self.left_img.caliLen / float(self.lineEdit.text())
                print('Calibration=', self.imgCalibration)
                self.logTextEdit.append("设置标尺：%.2f pixel/mm" % self.imgCalibration)
            else:
                QMessageBox.information(self, "ERROR", "标尺长度需大于0！", QMessageBox.Ok)

        elif self.tabWidget.currentIndex() == 1:
            if self.lineEdit.text() != '' and float(self.lineEdit.text()) > 0.0:
                self.vidCalibration = self.mainVid.caliLen / float(self.lineEdit.text())
                print('vidCalibration=', self.vidCalibration)
                self.vidLogTextEdit.append("设置标尺：%.2f pixel/mm" % self.vidCalibration)
            else:
                QMessageBox.information(self, "ERROR", "标尺长度需大于0！", QMessageBox.Ok)

    # 设置绘图范围响应
    def setPaintScope(self):
        self.paintScope = self.mainVid.scope
        print(self.paintScope)
        self.vidLogTextEdit.append("设置绘图范围：[x1=%.2f, x2=%.2f, y1=%.2f, y2=%.2f]" % tuple(self.paintScope))

    '''
    图像识别
    '''

    def sliderChange(self):
        self.picIouSpinBox.setValue(self.picIouSlider.value() / 100)
        self.picConfSpinBox.setValue(self.picConfSlider.value() / 100)

    def iouSpinBoxChange(self):
        self.picIouSlider.setValue(int(self.picIouSpinBox.value() * 100))
        self.iou = self.picIouSpinBox.value()

    def confSpinBoxChange(self):
        self.picConfSlider.setValue(int(self.picConfSpinBox.value() * 100))
        self.conf = self.picConfSpinBox.value()

    def labelStatus(self):
        self.hiddenLabels = not self.checkBox.isChecked()

    def confStatus(self):
        self.hiddenConf = not self.checkBox_2.isChecked()

    def lineWidthChange(self):
        self.lineWidth = self.detectLineWidthSpinBox.value()

    def inputSizeChange(self):
        self.lineEdit_3.setText(self.lineEdit_2.text())

    def clearAllResults(self):
        if self.diameters:
            self.diameters = []
            self.aveDiameter = 0.0
            self.label_6.setText('-')
            self.label_7.setText('-')
            self.label_18.setText('-')
            self.disFig.axes.clear()
            self.disFig.axes.set_xlabel('drop size(mm)')
            self.disFig.axes.set_ylabel('drop number')
            self.disFig.fig.canvas.draw_idle()

    def getPicSize(self):
        if self.img_path:
            self.lineEdit_2.setText(str(self.original_size[0]))
            self.lineEdit_3.setText(str(self.original_size[1]))

    def setDevice(self):
        self.device = "cuda:0" if self.gpuRadioButton.isChecked() else "cpu"
        print(self.device)

    def paintStatusChange(self):
        if self.freeRadioButton.isChecked():
            self.left_img.status = "free"
            self.mainVid.status = "free"
        elif self.verticalRadioButton.isChecked():
            self.left_img.status = "vertical"
            self.mainVid.status = "vertical"
        else:
            self.left_img.status = "horizontal"
            self.mainVid.status = "horizontal"

    def searchWeights(self):
        for fileName in listdir("weights/"):
            suffix = fileName.split('.')[-1]
            if suffix == "pt":
                self.detWeightsComboBox.addItem(fileName)
        self.det_weight = self.detWeightsComboBox.currentText()

    def selectWeights(self):
        self.det_weight = self.detWeightsComboBox.currentText()

    def withdrawResult(self):
        if self.currentNum == len(self.diameters):
            self.clearAllResults()
        elif self.currentNum:
            self.diameters = self.diameters[:len(self.diameters) - self.currentNum]
            self.label_18.setText('-')
            self.calcAveDia(self.diameters)

            self.disFig.axes.clear()
            self.disFig.axes.set_xlabel('drop size(mm)')
            self.disFig.axes.set_ylabel('drop number')
            self.disFig.axes.hist(self.diameters, edgecolor='k')
            self.disFig.fig.canvas.draw_idle()
            self.currentNum = 0

    # 结果导出
    def outputResults(self):
        if self.diameters:
            if not os.path.exists(f'{self.root}/results'):
                os.mkdir(f'{self.root}/results')
            creatTime = time.strftime("%Y-%m-%d-%Hh%Mmin%Ssec")
            with open(f'results/{creatTime}.csv', 'w') as f:
                f.write(f'{creatTime}\n\n')
                f.write('additional information:')
                addinfo = self.addTextEdit.toPlainText()
                if addinfo:
                    f.write(f'\n{addinfo}\n\n')
                else:
                    f.write('None\n\n')
                for index, dia in enumerate(self.diameters):
                    f.write(f'{index},{dia}\n')
            self.logTextEdit.append('导出成功！')

    def ch(self):
        print('change')

    def calcAveDia(self, diameters):
        thr = 0.0
        dub = 0.0
        for dia in diameters:
            thr += dia ** 3
            dub += dia ** 2
        sauAveDia = thr / dub
        self.label_7.setText("{:.2f}".format(sauAveDia))
        self.label_6.setText(str(len(diameters)))

    '''
    视频识别
    '''

    def setRealTimeDetect(self):
        if not self.realTimeDetect:
            self.realTimeDetect = True
            self.realTimedetBtn.setStyleSheet('QPushButton{background:yellow;}')
            self.rtDetStatelabel.setText('实时检测：开')
        else:
            self.realTimeDetect = False
            self.realTimedetBtn.setStyleSheet('')
            self.rtDetStatelabel.setText('实时检测：关')

    def setOcsortDetThr(self):
        self.ocsortDetThr = self.trackConfSpinBox.value()

    def setOcsortIouThr(self):
        self.ocsortIouThr = self.trackIouSpinBox.value()

    def setPauseStatus(self):
        if self.pauseFlag == False:
            self.pauseFlag = True
            self.pauseVidDetButton.setText('继续检测')
            self.pauseVidDetButton.setStyleSheet('QPushButton{background:yellow;}')
        elif self.pauseFlag == True:
            self.pauseFlag = False
            self.pauseVidDetButton.setText('暂停检测')
            self.pauseVidDetButton.setStyleSheet('')

    '''
    打开图片与视频、播放视频
    '''

    # 打开图片
    def upload_img(self):
        # 选择图片文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            print(fileName)
            self.isImg = True
            self.img_path = fileName
            im0 = imread(self.img_path)
            self.original_size = im0.shape[0], im0.shape[1]
            self.left_img.map = QPixmap(self.img_path).scaled(self.left_img.size(), Qt.KeepAspectRatio)
            self.left_img.setPixmap(self.left_img.map)
            self.logTextEdit.append("加载待检测图片：" + fileName)
            self.label_18.setText('-')
            print(self.left_img.width(), self.left_img.height())

    # 打开视频
    def upload_vid(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.vid_path = fileName
            self.vidLogTextEdit.append(f'打开视频：{self.vid_path}')
            self.rawVideo = cv2.VideoCapture(self.vid_path)
            ret, self.frame = self.rawVideo.read()
            if ret:
                video_img = frameToQImage(self.frame)
                self.mainVid.map = QPixmap.fromImage(video_img).scaled(self.mainVid.size(), Qt.KeepAspectRatio)
                self.mainVid.setPixmap(self.mainVid.map)
            # 视频进度条设置
            self.vidProgressHorizontalSlider.setValue(1)
            self.vidProgressHorizontalSlider.setMaximum(int(self.rawVideo.get(cv2.CAP_PROP_FRAME_COUNT)))
            # 获取帧率
            self.FPS = int(self.rawVideo.get(cv2.CAP_PROP_FPS))
            self.originalFrameSpinBox.setValue(self.FPS)
            self.outputFrameSpinBox.setValue(self.FPS)
            self.tracks = {}

    # 通过打开/关闭定时器来控制视频
    def startStopVideo(self):
        if self.rawVideo:
            if not self.isVideoPalying:
                # 假如视频播放完毕，从头开始
                if not self.rawVideo.isOpened():
                    self.rawVideo = cv2.VideoCapture(self.vid_path)
                # 设置视频播放状态为正在播放
                self.isVideoPalying = True
                self.playVidPushButton.setText('Stop')
                self.playVidPushButton.setStyleSheet('QPushButton{background:yellow;}')
                self.timerVideo.start(int(1000 / self.FPS))
                self.timerVideo.timeout.connect(self.openFrame)
            else:
                self.timerVideo.stop()
                self.playVidPushButton.setText('Start')
                self.playVidPushButton.setStyleSheet('')
                self.isVideoPalying = False

    def openFrame(self):
        """ Slot function to capture frame and process it
            """
        if self.rawVideo:
            ret, self.frame = self.rawVideo.read()
            if ret:
                video_img = frameToQImage(self.frame)
                self.mainVid.map = QPixmap.fromImage(video_img).scaled(self.mainVid.size(), Qt.KeepAspectRatio)
                self.mainVid.setPixmap(self.mainVid.map)
                self.vidProgressHorizontalSlider.setValue(int(self.rawVideo.get(cv2.CAP_PROP_POS_FRAMES)) + 1)
            else:
                self.rawVideo.release()
                self.timerVideo.stop()  # 停止计时器

    def setVideoProgressBar(self):
        if self.rawVideo:
            self.rawVideo.set(cv2.CAP_PROP_POS_FRAMES, self.vidProgressHorizontalSlider.value())
            ret, self.frame = self.rawVideo.read()
            if ret:
                video_img = frameToQImage(self.frame)
                self.mainVid.map = QPixmap.fromImage(video_img).scaled(self.mainVid.size(), Qt.KeepAspectRatio)
                self.mainVid.setPixmap(self.mainVid.map)

    def savePreFrame(self):

        if isinstance(self.frame, np.ndarray):
            frameDir = f'{self.root}/frames'
            if not os.path.exists(frameDir):
                os.mkdir(frameDir)
            print(f'{frameDir}/{self.vid_path}_frame_{int(self.rawVideo.get(cv2.CAP_PROP_POS_FRAMES)) + 1}.png')
            cv2.imwrite(
                f'{frameDir}/{self.vid_path.split("/")[-1]}_frame_{int(self.rawVideo.get(cv2.CAP_PROP_POS_FRAMES)) + 1}.png',
                self.frame)

    '''
    ***检测图片***
    '''

    # 图像检测函数
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img_path  # file/dir/URL/glob, 0 for webcam
        imgsz = [int(self.lineEdit_2.text()), int(self.lineEdit_3.text())]  # inference size (pixels)
        conf_thres = self.conf  # confidence threshold
        iou_thres = self.iou  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = self.lineWidth  # bounding box thickness (pixels)
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference

        self.statusbar.showMessage("正忙")

        if source == "":
            self.noImgInfo()
        elif self.imgCalibration == 0.0:
            self.caliInfo()
        elif not self.model:
            self.noMoelInfo()
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            self.picProgressBar.setValue(5)
            # Dataloader
            if webcam:
                view_img = check_imshow()
                # cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # Run inference
            if pt and device.type != 'cpu':
                model(zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                self.picProgressBar.setValue(20)
                for i, det in enumerate(pred):  # per image
                    # 写入当前图片液滴检测数
                    self.label_18.setText(str(len(det)))
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print('s=', s)

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh

                            # 根据标尺转换为实际直径（xywh为相对值）
                            wp = xywh[2] * self.left_img.map.width()  # 检测宽度在label上的pixel数(液滴尺寸为比例)
                            hp = xywh[3] * self.left_img.map.height()  # 检测高度在label上的pixel数
                            w = wp / self.imgCalibration  # 实际宽度
                            h = hp / self.imgCalibration  # 实际高度
                            # 计算椭圆等效直径
                            diameter = pow(h ** 2 * w, 1 / 3) if h <= w else pow(w ** 2 * h, 1 / 3)
                            self.currentNum = len(pred[0])
                            self.diameters.append(diameter)

                            if save_txt:  # Write to file
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = ''
                                if self.showLabelCheckBox.isChecked():
                                    label = label + f'{names[c]} '
                                if self.showConfCheckBox.isChecked():
                                    label = label + f'{conf:.2f} '
                                if self.showDiaCheckBox.isChecked():
                                    label = label + f'{diameter:.2f}mm'
                                    # label = label + f'{h:.2f}mm {w:.2f}mm'

                                annotator.box_label(xyxy, label, color=(0,255,0))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                #                  BGR=True)
                    # Print time (inference-only)
                    # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    self.num += 1
                    self.logTextEdit.append('## Detection ' + str(self.num) + " ## \n" + f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    resize_scale = output_size / im0.shape[0]
                    im0 = resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)

                    # 保存并显示图片
                    if not os.path.exists(self.imgSavePath):
                        os.makedirs(self.imgSavePath)
                    save_path = f"{self.imgSavePath}/{self.img_path.split('/')[-1]}"
                    imwrite(save_path, im0)
                    self.left_img.map = QPixmap(save_path).scaled(self.left_img.size(), Qt.KeepAspectRatio)
                    self.left_img.setPixmap(self.left_img.map)

            self.picProgressBar.setValue(100)

            # 绘制图像
            self.calcAveDia(self.diameters)
            self.disFig.axes.clear()
            self.disFig.axes.hist(self.diameters, edgecolor='w')
            self.disFig.axes.set_xlabel('drop size(mm)')
            self.disFig.axes.set_ylabel('drop number')
            self.disFig.fig.canvas.draw_idle()

            self.picProgressBar.setValue(0)
            self.statusbar.showMessage("就绪")

    '''
    ***检测视频***
    '''

    # 开启视频检测线程
    def detVideoThread(self):
        if not self.vid_path:
            self.noVidInfo()
        elif not self.model:
            self.noMoelInfo()
        else:
            self.vidDecThread = threading.Thread(target=self.det_vid)
            self.vidDecThread.start()

    # 强制结束当前检测线程
    # def stopVidDetThread(self):
    #     if self.vidDecThread:
    #         stop_thread(self.vidDecThread)
    #         self.vidLogTextEdit.append('已结束当前检测\n')

    # 提前结束当前检测
    def stopVidDetThread(self):
        if not self.stopVidDetFlag:
            self.stopVidDetFlag = True

    # 视频检测函数
    def det_vid(self):
        source = self.vid_path
        model = self.model
        yolo_weights = WEIGHTS / 'yolov5m.pt'  # model.pt path(s),
        strong_sort_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
        config_strongsort = ROOT / 'strong_sort/configs/strong_sort.yaml'
        imgsz = [int(self.lineEdit_2.text()), int(self.lineEdit_3.text())]  # inference size (height, width)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid = False  # show results
        save_txt = True  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        save_vid = True  # save confidences in --save-txt labels
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference
        visualize = False  # visualize features
        update = False  # update all models
        project = ROOT / 'runs/track'  # save results to project/name
        name = 'exp'  # save results to project/name
        exist_ok = False  # existing project/name ok, do not increment
        line_thickness = self.lineWidth  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        hide_class = False  # hide IDs
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        eval = False  # run multi-gpu eval

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        # if not isinstance(yolo_weights, list):  # single yolo model
        #     exp_name = yolo_weights.stem
        # elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        #     exp_name = Path(yolo_weights[0]).stem
        # else:  # multiple models after --yolo_weights
        #     exp_name = 'ensemble'
        # exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        save_dir = Path('res')
        # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # if eval:
        #     device = torch.device(int(device))
        # else:
        #     device = select_device(device)
        # model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            nr_sources = len(dataset)
        else:

            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # initialize StrongSORT
        # cfg = get_config()
        # cfg.merge_from_file(config_strongsort)

        # Create as many strong sort instances as there are video sources
        ocsort_list = []
        for i in range(nr_sources):
            ocsort_list.append(
                OCSort(
                    det_thresh=self.ocsortDetThr,
                    iou_threshold=self.ocsortIouThr,
                    use_byte=True
                )
            )
            # strongsort_list[i].model.warmup()
        outputs = [None] * nr_sources

        # Run tracking
        model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        # 储存上一次追踪结果
        pre_outputs = None
        rlRatio = 0.0

        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                curr_frames[i] = im0
                # print(im0.shape,'imooooooooooooooo')
                # txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                txt_path = str(save_dir / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop

                annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)

                # if cfg.STRONGSORT.ECC:  # camera motion compensation
                #    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort
                    t4 = time_sync()
                    outputs[i] = ocsort_list[i].update(det.detach().cpu().numpy())
                    t5 = time_sync()
                    dt[3] += t5 - t4
                    # print(outputs)
                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            # cls = output[5]
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]

                            # 计算原图到label的比例（label中一个pixel对应原图中多少个pixel）
                            if rlRatio == 0.0:
                                rlRatio = 0.5 * ((im0.shape[0] / self.mainVid.map.height()) + (
                                        im0.shape[1] / self.mainVid.map.width()))
                                self.vidCalibration *= rlRatio

                            # 计算液滴尺寸
                            w = bbox_w / self.vidCalibration  # 实际宽度
                            h = bbox_h / self.vidCalibration  # 实际高度
                            diameter = pow(h ** 2 * w, 1 / 3) if h <= w else pow(w ** 2 * h, 1 / 3)  # 液滴尺寸mm
                            self.vidDiameters.append(diameter)

                            # 以像素表示的液滴中心坐标
                            center = (bbox_left + 0.5 * bbox_w, bbox_top + 0.5 * bbox_h)
                            ic = (int(center[0]),int(center[1]))

                            # 将中心坐标加入轨迹队列
                            if id not in self.tracks:
                                a = []
                                self.tracks[id] = []
                                a.append(ic)
                            else:
                                if len(self.tracks[id])==self.trackLength:
                                    self.tracks[id].pop(0)
                                    self.tracks[id].append(ic)
                                else:
                                    self.tracks[id].append(ic)

                            # 绘制轨迹
                            track = self.tracks[id]
                            if len(track)>=2:
                                # print(track)
                                for k in range(len(track)-1):
                                    cv2.line(im0,track[k],track[k+1],(0,255,0),2)


                            distance = -1
                            # 遍历前一帧数据
                            if isinstance(pre_outputs, np.ndarray):
                                for pre_output in pre_outputs:
                                    if pre_output[-1] == id:

                                        pre_center = (pre_output[0] + 0.5 * (pre_output[2] - pre_output[0]),
                                                      pre_output[1] + 0.5 * (pre_output[3] - pre_output[1]))

                                        distance = np.sqrt((center[0] - pre_center[0]) ** 2 + (
                                                center[1] - pre_center[1]) ** 2)  # 两帧单液滴间距 pixel（原图）

                                        x = (center[0] / im0.shape[1] - self.paintScope[0]) / (
                                                self.paintScope[1] - self.paintScope[0])
                                        y = (center[1] / im0.shape[0] - self.paintScope[2]) / (
                                                self.paintScope[3] - self.paintScope[2])
                                        vx = (center[0] - pre_center[0]) / im0.shape[1] / (
                                                self.paintScope[1] - self.paintScope[0])
                                        vy = (center[1] - pre_center[1]) / im0.shape[0] / (
                                                self.paintScope[3] - self.paintScope[2])

                                        if 0 < x < 1 and 0 < y < 1:
                                            self.arrowDatas.append([x, 1 - y, vx, -vy])
                                        # self.arrowDatas.append([x,1-y,0.05,0.05])

                            realDistance = distance / self.vidCalibration / 1000  # 两帧单液滴实际间距 m
                            velocity = realDistance / (1.0 / self.originalFrameSpinBox.value())  # 单液滴速度 m/s

                            if save_txt:
                                # to MOT format

                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 12 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, -1, distance,
                                                                   i))

                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                # c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = ''
                                if self.showIdCheckBox.isChecked():
                                    label += str(id)
                                if self.showVelocityCheckBox.isChecked():
                                    if velocity < 0:
                                        pass
                                    else:
                                        label += f'{velocity:.4f}m/s'
                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), OCSORT:({t5 - t4:.5f}s)')
                    self.vidLogTextEdit.append(f'{s}Done. YOLO:({t3 - t2:.3f}s), OCSORT:({t5 - t4:.5f}s)')
                    pre_outputs = outputs[i]

                else:
                    # strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')
                    pre_outputs = None

                # Stream results
                im0 = annotator.result()

                # 设置实时检测
                if self.realTimeDetect:
                    rtImg = frameToQImage(im0)
                    self.mainVid.map = QPixmap.fromImage(rtImg).scaled(self.mainVid.size(), Qt.KeepAspectRatio)
                    self.mainVid.setPixmap(self.mainVid.map)

                # 绘制动态液滴分布图像
                if self.checkBox_2.isChecked():
                    self.vidDisFig.axes.clear()
                    self.vidDisFig.axes.set(xlim=[0, 10], ylim=[0, 30],
                                            ylabel='drop number', xlabel='drop size(mm)')
                    self.vidDisFig.axes.hist(self.vidDiameters, edgecolor='w', color='orange')
                    self.vidDisFig.fig.canvas.draw_idle()
                    self.vidDiameters.clear()

                # 绘制速度矢量图像
                if self.checkBox.isChecked():
                    if self.arrowDatas:
                        self.arrowFig.axes.clear()
                        # print(self.arrowDatas)
                        data = np.array(self.arrowDatas)
                        self.arrowFig.axes.set(xlim=[0, 1], ylim=[0, 1])
                        self.arrowFig.axes.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], color='b')
                        self.arrowFig.axes.scatter(data[:, 0], data[:, 1], color='w', marker='o', edgecolors='g', s=200) # color = ''  ??
                        self.arrowFig.fig.canvas.draw_idle()
                        self.arrowDatas.clear()

                # Save results (image with detections)
                if save_vid:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        fps = self.outputFrameSpinBox.value()
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]

            # 使用循环暂停检测
            while self.pauseFlag:
                if self.pauseFlag == False:
                    break
                time.sleep(0.1)

            # 提前结束检测
            if self.stopVidDetFlag:
                self.stopVidDetFlag = False
                self.tracks = {}
                break

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms OC sort update per image at shape {(1, 3, *imgsz)}' % t)
        self.vidLogTextEdit.append(
            '----------------------------------------------------------------------------------------------')
        self.vidLogTextEdit.append(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms OC sort update per image at shape {(1, 3, *imgsz)}' % t)

        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    app = QApplication(argv)
    mainWindow = MainWindow()
    mainWindow.show()
    exit(app.exec_())
