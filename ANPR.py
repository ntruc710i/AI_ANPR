# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ANPR.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1081, 599)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(30, 30, 641, 461))
        self.image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(810, 420, 201, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.BSLB = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.BSLB.setFont(font)
        self.BSLB.setFrameShape(QtWidgets.QFrame.Box)
        self.BSLB.setText("")
        self.BSLB.setAlignment(QtCore.Qt.AlignCenter)
        self.BSLB.setObjectName("BSLB")
        self.horizontalLayout.addWidget(self.BSLB)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(230, 510, 241, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.control_bt = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.control_bt.setObjectName("control_bt")
        self.horizontalLayout_2.addWidget(self.control_bt)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(1)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.openimg = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.openimg.setObjectName("openimg")
        self.horizontalLayout_2.addWidget(self.openimg)
        self.detect = QtWidgets.QPushButton(self.centralwidget)
        self.detect.setGeometry(QtCore.QRect(830, 530, 151, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.detect.setFont(font)
        self.detect.setObjectName("detect")
        self.BSLB_2 = QtWidgets.QLabel(self.centralwidget)
        self.BSLB_2.setGeometry(QtCore.QRect(850, 350, 131, 59))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.BSLB_2.setFont(font)
        self.BSLB_2.setLocale(QtCore.QLocale(QtCore.QLocale.Vietnamese, QtCore.QLocale.Vietnam))
        self.BSLB_2.setAlignment(QtCore.Qt.AlignCenter)
        self.BSLB_2.setObjectName("BSLB_2")
        self.lb_bienso = QtWidgets.QLabel(self.centralwidget)
        self.lb_bienso.setGeometry(QtCore.QRect(790, 150, 231, 161))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lb_bienso.setFont(font)
        self.lb_bienso.setAutoFillBackground(True)
        self.lb_bienso.setFrameShape(QtWidgets.QFrame.Box)
        self.lb_bienso.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lb_bienso.setText("")
        self.lb_bienso.setObjectName("lb_bienso")
        self.lb_namebs = QtWidgets.QLabel(self.centralwidget)
        self.lb_namebs.setGeometry(QtCore.QRect(790, 98, 231, 51))
        self.lb_namebs.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lb_namebs.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lb_namebs.setText("")
        self.lb_namebs.setObjectName("lb_namebs")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.control_bt.setText(_translate("MainWindow", "Start"))
        self.openimg.setText(_translate("MainWindow", "Open Picture"))
        self.detect.setText(_translate("MainWindow", "Detect"))
        self.BSLB_2.setText(_translate("MainWindow", "BIỂN SỐ"))