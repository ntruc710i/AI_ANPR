
from fileinput import filename
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
import os
import numpy as np
import sys
from His import Ui_MainWindowx
class LoadQt(QMainWindow):
    filename = ""
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindowx()
        self.ui.setupUi(self)

        self.model = QtGui.QStandardItemModel()   # <----
        self.ui.listView.setModel(self.model) 
        
        f = open('Data_input.txt')
        for line in f:
            it = QtGui.QStandardItem(str(line))
            self.model.appendRow(it)
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = LoadQt()
    mainWindow.show()

    sys.exit(app.exec_())