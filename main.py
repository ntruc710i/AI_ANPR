from distutils.log import error
from fileinput import filename
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi

import sys
import cv2
from ANPR import *

import numpy as np
from txtdata import data_out_txt
from lib_detection import load_model, detect_lp, im2single

class LoadQt(QMainWindow):
    filename = ""
    sizeimg=(641,461)
    sizebs=(231,161)
    namebs=cv2.imread("image/namebs.jpg")
    logobs=cv2.imread("image/logobienso.jpg")
    logomain=cv2.imread("image/nhandienbs.jpg")
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)
        self.ui.openimg.clicked.connect(self.open_img)
        self.ui.detect.clicked.connect(self.detect_image)
        self.showimage(self.namebs,(231,51),self.ui.lb_namebs)
        self.showimage(self.logobs,self.sizebs,self.ui.lb_bienso)
        self.showimage(self.logomain,self.sizeimg,self.ui.image_label)
    

    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        
    

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
            # if timer is started
        else:
            # stop timer
            ret, image = self.cap.read()
            self.timer.stop()
            # release video capture
            self.cap.release()
            #save image
            showPic = cv2.imwrite("filename1.jpg",image)
            self.filename="filename1.jpg"
            self.showimage(self.logobs,self.sizebs,self.ui.lb_bienso)
            # update control_bt text
            self.ui.control_bt.setText("Start")

            
            
    
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.filename=fname
        self.tmp = self.image
        self.showimage(self.image,self.sizeimg,self.ui.image_label)

        self.showimage(self.logobs,self.sizebs,self.ui.lb_bienso)
    
    """def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()
        self.ui.image_label.setPixmap(QPixmap.fromImage(img))
        self.ui.image_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)"""
    #Hiển thị ảnh lên GUI
    def showimage(self,image,size,img_lb):
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        resized = cv2.resize(image, size)
        # get image infos
        height, width, channel = resized.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(resized.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        img_lb.setPixmap(QPixmap.fromImage(qImg))
    #Mở hình ảnh
    def open_img(self):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File', '/Project_ANPR/test/', "Image Files (*)")
        #print(fname)
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")
    # Nhận diện biển số bằng hình ảnh
    def detect_image(self):
        #self.ui.openimg.clicked.connect(self.open_img)
       
        # Ham sap xep contour tu trai sang phai
        def sort_contours(cnts):

            reverse = False
            i = 0
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][i], reverse=reverse))
            return cnts

        # Dinh nghia cac ky tu tren bien so
        char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

        # Ham fine tune bien so, loai bo cac ki tu khong hop ly
        def fine_tune(lp):
            newString = ""
            for i in range(len(lp)):
                if lp[i] in char_list:
                    newString += lp[i]
            return newString

        #while True :
        img_path = self.filename
        # Load model LP detection
        wpod_net_path = "wpod-net_update1.json"
        wpod_net = load_model(wpod_net_path)
        # Đọc file ảnh đầu vào
        Ivehicle = cv2.imread(img_path)

        try:
            Ivehicle.shape
            print("checked for shape".format(Ivehicle.shape))
        except AttributeError:
            print("shape not found")
        #print (Ivehicle)
        # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
        Dmax = 608
        Dmin = 288

        # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)

        try:
            _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
            print(lp_type)
        except :
            print("bien so null")  
            self.ui.BSLB.setText("NULL")        
            #break
            #quit()
        try :
                # Cau hinh tham so cho model SVM
                digit_w = 30 # Kich thuoc ki tu
                digit_h = 60 # Kich thuoc ki tu

                model_svm = cv2.ml.SVM_load('svm.xml')

                
                if (len(LpImg) and lp_type==1):                

                    # Chuyen doi anh bien so
                    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

                    roi = LpImg[0]

                    # Chuyen anh bien so ve gray
                    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


                    # Ap dung threshold de phan tach so va nen
                    binary = cv2.threshold(gray, 127, 255,
                                cv2.THRESH_BINARY_INV)[1]

                    #cv2.imshow("Anh bien so sau threshold", binary)
                    #cv2.waitKey()

                    # Segment kí tự
                    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
                    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


                    plate_info = ""

                    for c in sort_contours(cont):
                        (x, y, w, h) = cv2.boundingRect(c)
                        ratio = h/w
                        if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                            if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                                # Ve khung chu nhat quanh so
                                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                # Tach so va predict
                                curr_num = thre_mor[y:y+h,x:x+w]
                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                                curr_num = np.array(curr_num,dtype=np.float32)
                                curr_num = curr_num.reshape(-1, digit_w * digit_h)
                                    

                                # Dua vao model SVM

                                result = model_svm.predict(curr_num)[1]
                                result = int(result[0, 0])

                                if result<=9: # Neu la so thi hien thi luon
                                        result = str(result)
                                else: #Neu la chu thi chuyen bang ASCII
                                    result = chr(result)

                                plate_info +=result

                    #cv2.imshow("Cac contour tim duoc", roi)
                    #cv2.waitKey()

                    # Viet bien so len anh
                    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

                    #Luu bien so va thoi gian vao file txt
                    data_out_txt(plate_info)

                    # Hien thi anh
                    print("Bien so=", plate_info)
                    #cv2.imshow("Hinh anh output",Ivehicle)
                    #cv2.waitKey()
                    #hình ảnh biển số
                    self.showimage(roi,self.sizebs,self.ui.lb_bienso)
                    self.ui.BSLB.setText(plate_info)
                else :
                    if (len(LpImg) and lp_type==2):
                        #print("Bien 2 dong 1")
                        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
                            
                        grayV = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
                        binaryV = cv2.threshold(grayV, 127, 255,
                            cv2.THRESH_BINARY_INV)[1]
                            
                        plate_info = ""
                        x=0
                        y=0
                        w=binaryV.shape[0]-100
                        h=binaryV.shape[1]
                        crop1=binaryV[y:w,x:h]
                        roiV = LpImg[0]
                        #cv2.imshow("Anh chia dong 1",binaryV)
                        #cv2.waitKey()
                        kernel3_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        thre_mor1 = cv2.morphologyEx(crop1, cv2.MORPH_DILATE, kernel3_1)
                        cont1, _  = cv2.findContours(thre_mor1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                        for c in sort_contours(cont1):
                            (x, y, w, h) = cv2.boundingRect(c)
                            ratio = h/w
                            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                                if h/roiV.shape[0]>=0.3: # Chon cac contour cao tu 60% bien so tro len

                                    # Ve khung chu nhat quanh so
                                    #cv2.rectangle(roiV, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                        # Tach so va predict
                                    curr_num = thre_mor1[y:y+h,x:x+w]
                                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                                    curr_num = np.array(curr_num,dtype=np.float32)
                                    curr_num = curr_num.reshape(-1, digit_w * digit_h)
                                    

                                    # Dua vao model SVM

                                    result = model_svm.predict(curr_num)[1]
                                    result = int(result[0, 0])

                                    if result<=9: # Neu la so thi hien thi luon
                                        result = str(result)
                                    else: #Neu la chu thi chuyen bang ASCII
                                        result = chr(result)

                                    plate_info +=result
                        #cv2.imshow("Anh bien so sau threshold 2 dong ", binaryV)
                        #cv2.waitKey()
                        #cv2.imshow("Cac contour tim duoc 1", roiV)
                        #cv2.waitKey()


                        crop2=binaryV[100:200,0:280]
                        #cv2.imshow("Anh chia dong 2",crop2)
                        #cv2.waitKey()
                        roiV1 = LpImg[0]
                        kernel3_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        thre_mor2 = cv2.morphologyEx(crop2, cv2.MORPH_DILATE, kernel3_2)
                        cont2, _  = cv2.findContours(thre_mor2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                        for c in sort_contours(cont2):
                            (x, y, w, h) = cv2.boundingRect(c)
                            ratio = h/w
                            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                                if h/roiV1.shape[0]>=0.3: # Chon cac contour cao tu 60% bien so tro len

                                    # Ve khung chu nhat quanh so
                                    #cv2.rectangle(roiV1, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                    # Tach so va predict
                                    curr_num = thre_mor2[y:y+h,x:x+w]
                                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                                    curr_num = np.array(curr_num,dtype=np.float32)
                                    curr_num = curr_num.reshape(-1, digit_w * digit_h)
                                    

                                    # Dua vao model SVM

                                    result = model_svm.predict(curr_num)[1]
                                    result = int(result[0, 0])

                                    if result<=9: # Neu la so thi hien thi luon
                                            result = str(result)
                                    else: #Neu la chu thi chuyen bang ASCII
                                        result = chr(result)

                                    plate_info +=result

                        #cv2.imshow("Cac contour tim duoc 2", roiV1)
                        print("Bien so:",plate_info)
                        #Luu bien so va thoi gian vao file txt
                        data_out_txt(plate_info)
                        cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
                        #cv2.imshow("Hinh anh output",Ivehicle)
                        #cv2.waitKey()

                        #cv2.imshow("anh abc",roiV)
                        #print("Bien 2 dong 2")
                        #self.loadImageBS(Ivehicle)
                        #Hình ảnh biển số
                        self.showimage(roiV,self.sizebs,self.ui.lb_bienso)
                        self.ui.BSLB.setText(plate_info)
                        
        except:
            print("------")
            print("Có ngoại lệ ",sys.exc_info()[0]," xảy ra.")
            self.ui.BSLB.setText("NULL")

        

        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = LoadQt()
    mainWindow.show()

    sys.exit(app.exec_())