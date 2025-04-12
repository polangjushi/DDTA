import cv2
from IPython.external.qt_for_kernel import QtGui
from PyQt5.QtCore import QRect, QPoint, QSize, Qt
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtWidgets import QLabel
import ctypes
import inspect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time


# 重写QLabel，用于图像和视频显示
class MyLabel(QLabel):
    map = None
    flag = False
    caliLen = 0.0
    scope = [.0, 1.0, .0, 1.0]
    scaleRes = False
    scopeRes = False
    scaleData = [0, 0, 0, 0]
    scopeData = [0, 0, 0, 0]
    scale = 1
    status = "free"  # 绘制状态（1，自由；2，垂直；3，水平）

    # 鼠标点击事件
    def mousePressEvent(self, event):
        if self.scaleRes:
            if event.button() == Qt.LeftButton:
                self.flag = True
                self.scaleData[0] = event.x()
                self.scaleData[1] = event.y()
            if event.button() == Qt.RightButton:
                self.scaleData = [0, 0, 0, 0]
                self.update()

        elif self.scopeRes:
            if event.button() == Qt.LeftButton:
                self.flag = True
                self.scopeData[0] = event.x()
                self.scopeData[1] = event.y()
            if event.button() == Qt.RightButton:
                self.scopeData = [0, 0, 0, 0]
                self.update()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag and self.scaleRes:
            self.scaleData[2] = event.x()
            self.scaleData[3] = event.y()
            self.update()

        elif self.flag and self.scopeRes:
            self.scopeData[2] = event.x()
            self.scopeData[3] = event.y()
            self.update()

    def wheelEvent(self, event):
        pass

    # 绘制事件
    def paintEvent(self, event):
        super(MyLabel, self).paintEvent(event)
        painter = QPainter(self)
        painter.begin(self)
        if self.scaleRes == True:

            painter.setPen(QPen(Qt.green, 1, Qt.SolidLine))
            if self.scaleData != [0, 0, 0, 0]:
                if self.status == "vertical":
                    self.scaleData[2] = self.scaleData[0]
                elif self.status == "horizontal":
                    self.scaleData[3] = self.scaleData[1]
                painter.drawRect(QRect(QPoint(self.scaleData[0] - 5, self.scaleData[1] - 5), QSize(10, 10)))
                painter.drawLine(self.scaleData[0], self.scaleData[1], self.scaleData[2], self.scaleData[3])
                painter.drawRect(QRect(QPoint(self.scaleData[2] - 5, self.scaleData[3] - 5), QSize(10, 10)))

                self.caliLen = ((self.scaleData[2] - self.scaleData[0]) ** 2 + (
                        self.scaleData[3] - self.scaleData[1]) ** 2) ** 0.5

                # width, height
                self.cali = (abs(self.scaleData[2] - self.scaleData[0]) / self.width(),
                             abs(self.scaleData[3] - self.scaleData[1]) / self.height())
                # print(self.width(), self.height())
                # print(self.cali)
                # print(self.caliLen)

        if self.scopeRes == True:

            painter.setPen(QPen(Qt.yellow, 1, Qt.SolidLine))
            if self.scopeData != [0, 0, 0, 0]:
                width = self.scopeData[2] - self.scopeData[0]
                height = self.scopeData[3] - self.scopeData[1]
                painter.drawRect(QRect(QPoint(self.scopeData[0] - 5, self.scopeData[1] - 5), QSize(10, 10)))
                painter.drawRect(self.scopeData[0], self.scopeData[1], width, height)
                painter.drawRect(QRect(QPoint(self.scopeData[2] - 5, self.scopeData[3] - 5), QSize(10, 10)))
                self.scope = self.scopeData[0] / self.width(), self.scopeData[2] / self.width(),self.scopeData[1] / self.height(), self.scopeData[3] / self.height()
                # print("scope:", self.scope)
        painter.end()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self.map != None:
            pass
            # self.clear()
            # time.sleep(0.1)
            self.map = self.map.scaled(self.size(), Qt.KeepAspectRatio)
            self.setPixmap(self.map)
            print(self.map.size())
            print(self.size(), '----')


# 创建绘图类
class MyFigure(FigureCanvas):
    def __init__(self):
        self.fig = Figure()  # 设置长宽以及分辨率
        super(MyFigure, self).__init__(self.fig)
        # self.ax = self.fig.add_subplot(111)  # 创建axes对象实例，这个也可以在具体函数中添加
        # self.axes = self.fig.add_axes([0.12, 0.1, 0.8, 0.85])


# 强制结束线程
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


# 将 ndarray 转换为 QPixmap

def frameToQImage(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
    return vedio_img

# def frameToQImage(frame):
#     if len(frame.shape) == 3:
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
#     elif len(frame.shape) == 1:
#         vedio_img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_Indexed8)
#     else:
#         vedio_img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.Format_RGB888)
#     return vedio_img
