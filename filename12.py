# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camerapaper2.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFileDialog, QApplication,QMainWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
import cv2, imutils
import os
from JetsonCamera import Camera
#from cv2 import imshow
#from threading import Thread, local
import nanocamera as nano
import subprocess
#import RPi.GPIO as R_GPIO
import Jetson.GPIO as GPIO
from time import sleep
import numpy as np
import mysql.connector
from mysql.connector import Error
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ConnectionException
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
ip_address = '192.168.4.20'
#client = ModbusTcpClient(ip_address)

pin_out = [11,13,15,19,21,23]
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
pin_out = [11,13,15,19,21,23]
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
#GPIO.setup(15,GPIO.OUT)
GPIO.setup(19,GPIO.OUT)
GPIO.setup(21,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
set_data=[]
def set_camera_control(control,value):

    command= f"v4l2-ctl -d /dev/video0 --set-ctrl={control}={value}"
    try:
        subprocess.run(command,shell=True,check =True)
        print("OK")
    except subprocess.CalledProcessError as e:
        print("NG")
set_camera_control("exposure_auto",1)
set_camera_control("exposure_absolute",1200)
    
#from pymodbus.client.sync import ModbusTcpClient
#client = ModbusTcpClient('192.168.4.28')
#result = client.read_holding_registers(0,1)
# client.write_register(0,1234)
#client.close()
#net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'ssd.caffemodel')
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#net = cv2.dnn.readNetFromONNX('best.onnx')
# Load the labels for the model
labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
          11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
          16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

class globalvar:
    read_database=[]
    pictureid=0
    runmodel=0
    qtycam1 =1
    qtycam2=1
    save_picture_1=0
    save_picture_2 =0
class camera1(QThread):
    finished = pyqtSignal()
    camera1 = pyqtSignal(QImage)
    anacamera1 = pyqtSignal(QImage)
    send_actualpixel1 = pyqtSignal(int)
    send_actualdifferent1=pyqtSignal(int)
    send_contour1 =pyqtSignal(int)
    send_actuallengh1 =pyqtSignal(int)
        #self.thread1 = camera1(send_database,self.pictureid,self.runmodel,self.qtycam1,save_picture_1)
        #self.thread1.camera1.connect(self.show_cam1)
        #self.thread1.anacamera1.connect(self.show_anacamera1)
        #self.thread1.send_actualpixel1.connect(self.show_actualpixel1)
        #self.thread1.send_actualdifferent1.connect(self.show_actualdifferent1) 
        #self.thread1.send_contour1.connect(self.show_contour1)
        #self.thread1.send_actuallengh1.connect(self.show_actuallengh1)
    def __init__(self,read_database,pictureid,runmodel,qtycam1,save_picture_1):
        super().__init__()
        globalvar.read_database=read_database
        globalvar.pictureid=pictureid
        globalvar.runmodel=runmodel
        globalvar.qtycam1 = qtycam1
        globalvar.save_picture_1 = save_picture_1
        self.kernel = np.ones((3,3),np.uint8)       
    def stackImages(self,scale,imgArray):    
        rows = len(imgArray)
        #print(rows)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0],list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0,rows):
                for y in range(0,cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)

                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height,width,3),np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0,rows):
                hor[x] =np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x],(0,0),None,scale,scale)
                    #imgArray[x] = cv2.resize(imgArray[x], (500, 400))
                else:
                    imgArray[x] = cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)
                    #imgArray[x] = cv2.resize(imgArray[x], (500, 400))
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)


            hor = np.hstack(imgArray)
            ver = hor
        return ver
    def load_data(self):        
        connection = mysql.connector.connect(host='localhost',
                                            database = 'CAMERA_PAPER',
                                            user ='vqbg',
                                            password='thanhtam2408V!@')
        
        
        mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA_2 where Model = %s"
        val_select=(int(self.runmodel),)
        cursor = connection.cursor()
        result_model = cursor.execute(mySql_Select_Table_Query,val_select)
        rows=cursor.fetchall()
        row_count=cursor.fetchmany()
        set_data = np.array(rows)
        cursor.close()
        connection.close()
        return set_data

    def run(self):
        #cap = nano.Camera(device_id=0,flip=0,width= 640,height= 480, fps= 30)
        cap = nano.Camera(camera_type =1,device_id=0,width= 640,height= 480, fps= 30)
        M200=0
        while True:
            #set_data=self.load_data()

            #M200_bit = client.read_coils(200, 1)  # nhận tín hiệu cylinder cắt đi lên
            #if M200_bit.isError():
                #print("Error reading coil")
            #else:
                # coil_value_Y0 = result_Y0.bits[0]  # Lấy trạng thái của coil tại địa chỉ 0
                # 3print(f'Trạng thái của coil 00001: {coil_value_Y0}')
                #M200 = M200_bit.bits[0]
            # if M200==0:
            # M300=1 trạng thái chưa cho tín hiệu
            if M200 == 1:
                # set_camera_control("exposure_auto",1)
                # set_camera_control("exposure_absolute",10)
                cap = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30)
                # sleep(1)
            # if capsized():
            #if cap.isReady():
            if M200 == 0  and cap.isReady():
                self.cap_out = cap.read()               
                shapes_cap_out = np.zeros_like(self.cap_out,np.uint8)




                for r in range(globalvar.qtycam1):
                    #self.read_database[r,2]: upperframe ;self.read_database[r,3]: lowwer frame
                    #self.read_database[r,4]: leftframe ;self.read_database[r,5]: right frame
                    height, width, channels = shapes_cap_out.shape
                    mask_outside = np.zeros((height, width), dtype=np.uint8)
                    mask_inside = np.zeros((height, width), dtype=np.uint8)
                    # axis_major_outside, axis_minor_outside = int(globalvar.read_database[r,50]), int(globalvar.read_database[r,51])
                    # axis_major_inside, axis_minor_inside = int(globalvar.read_database[r,53]), int(globalvar.read_database[r,54])
                    xcenter, ycenter = int(globalvar.read_database[r, 50]), int(globalvar.read_database[r, 51])
                    r1, r2 = int(globalvar.read_database[r, 53]), int(globalvar.read_database[r, 54])
                    cv2.ellipse(mask_outside, (int(xcenter), int(ycenter)), (r1, r1), globalvar.read_database[r, 52], 0,
                                360, (255, 255, 255), -1)
                    cv2.ellipse(mask_inside, (int(xcenter), int(ycenter)), (r2, r2), globalvar.read_database[r, 55], 0,
                                360,
                                (255, 255, 255), -1)
                    # taoj ring mau trang
                    mask_ring = cv2.bitwise_xor(mask_inside, mask_outside)
                    image_home = cv2.bitwise_and(self.cap_out, mask_ring)

                    # inside_inv = cv2.bitwise_not(mask_inside)

                    # inside= cv2.bitwise_and(imgCropped_1, imgCropped_1, mask=inside_inv)
                    # outside = cv2.bitwise_and(inside, inside, mask=mask_outside)
                    # outside_inv = cv2.bitwise_not(mask_outside)

                    # mask_all = cv2.bitwise_or(outside_inv,mask_inside)
                    # imgCropped_1 = cv2.bitwise_not(outside, outside, mask=mask_all)

                    cv2.rectangle(shapes_cap_out ,(globalvar.read_database[r,6],globalvar.read_database[r,4]),(globalvar.read_database[r,7],globalvar.read_database[r,5]),(255,0,0),4)
                    imgCropped= self.cap_out[globalvar.read_database[r,4]:globalvar.read_database[r,5],globalvar.read_database[r,6]:globalvar.read_database[r,7]]
                    alpha_b = (globalvar.read_database[r,29])/10
                    brightness_img = cv2.convertScaleAbs(imgCropped, alpha = alpha_b, beta = 0)
                    imgCropped = cv2.convertScaleAbs(imgCropped, alpha = alpha_b, beta = 0)

                    contrast_factor = (globalvar.read_database[r,30])/10
                    contrasted_img = cv2.addWeighted(imgCropped,contrast_factor,np.zeros(imgCropped.shape,imgCropped.dtype),0,0)
                    imgCropped = cv2.addWeighted(imgCropped,contrast_factor,np.zeros(imgCropped.shape,imgCropped.dtype),0,0)
                    
                    read_image = np.zeros_like(imgCropped,np.uint8)
                    diff = np.zeros_like(imgCropped,np.uint8)
                    diffBinary = np.zeros_like(imgCropped,np.uint8)
                    
                    imgGray = cv2.cvtColor(imgCropped,cv2.COLOR_BGR2GRAY)
                    imgBlur = cv2. GaussianBlur(imgGray,(3,3),0)
                    #self.read_database[r,10] : adjust threshold
                    imgBinary= cv2.threshold(imgBlur,globalvar.read_database[r,10],255,cv2.THRESH_BINARY)[1]
                    imgBinary = cv2.erode(imgBinary,self.kernel, iterations=globalvar.read_database[r,32])
                    imgBinary = cv2.dilate(imgBinary,self.kernel, iterations=globalvar.read_database[r,31])
                    actualpixel1 = cv2.countNonZero(imgBinary)
                    
                    file_path = 'picture_{}.jpg'
                    if os.path.exists(file_path.format(globalvar.pictureid)):
                        read_image = cv2.imread('picture_{}.jpg'.format(globalvar.pictureid))
                        if imgCropped.shape != read_image.shape:
                            read_image = cv2.resize(read_image,(imgCropped.shape[1],imgCropped.shape[0]))
                        gray1 = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(gray1, gray2)
                        #self.read_database[r,13]:adjust different
                        diffbinary= cv2.threshold(diff,globalvar.read_database[r,13],255,cv2.THRESH_BINARY)[1]
                        actualdifferent1 = cv2.countNonZero(diffbinary)
                    else:
                        print("hay luu anh")
                    imgstack = self.stackImages(1,([imgCropped,imgBinary],[image_home ,mask_ring] ))
                  
                    # contour_up,hierarchy = cv2.findContours(imgBinary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    # contour_2,hierarchy = cv2.findContours(imgBinary_2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    # cnt_number = len(contour_up)
                    # cnt_number_2 = len(contour_2)
                    # sum_up = 0
                    # area_up = 0
                    # sum_2 = 0
                    # area_2 = 0
                    
                    # for cnt_up in contour_up:
                    #     area_up = cv2.contourArea(cnt_up)
                    #     sum_up=sum_up+area_up
                    #     #c_up =c_up+1
                    # Địa chỉ IP của thiết bị
                    #Coils (đọc/ghi): Địa chỉ từ 00001 đến 09999 (bit). Y,M,T<C,...
                    #Discrete Inputs (chỉ đọc): Địa chỉ từ 10001 đến 19999 (bit). X
                    #Input Registers (chỉ đọc): Địa chỉ từ 30001 đến 39999 (16-bit).Only Read D0-->
                    #Holding Registers (đọc/ghi): Địa chỉ từ 40001 đến 49999 (16-bit). Read/WriteD0-->

                    if (actualpixel1> globalvar.read_database[r,12]):
                        if r == 0 and client.connect():
                            GPIO.output(11,GPIO.HIGH)
                            client.write_coil(1, True)
                            cv2.putText(shapes_cap_out ,"11ON-> X10 On",(globalvar.read_database[r,6],globalvar.read_database[r,4]),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2)
                        elif r==1:
                            GPIO.output(13,GPIO.HIGH)
                            cv2.putText(shapes_cap_out ,"13ON--> X11 ON",(globalvar.read_database[r,6],globalvar.read_database[r,4]),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,0),2)
                        else:
                            a=1  
                    else:
                        if r == 0:
                            GPIO.output(11,GPIO.LOW)
                            cv2.putText(shapes_cap_out ,"11OFF-> X10OFF",(globalvar.read_database[r,6],globalvar.read_database[r,4]),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2)
                        elif r==1:
                            GPIO.output(13,GPIO.LOW)
                            cv2.putText(shapes_cap_out ,"13OFF-> X11OFF",(globalvar.read_database[r,6],globalvar.read_database[r,4]),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2)
                        else:
                            a=0                      
                    self.save_pic_ok= 1   
                    if globalvar.read_database[r,1] == globalvar.pictureid:
                        filename = 'picture_{}.jpg'
                        if self.save_pic_ok == globalvar.save_picture_1:
                            cv2.imwrite(filename.format(globalvar.pictureid), imgCropped)
                        #send_actualpixel1 = pyqtSignal(int)
                        #send_actualdifferent1=pyqtSignal(int)
                        #send_contour1 =pyqtSignal(int)
                        #send_actuallengh1 =pyqtSignal(int)
                        self.send_actualpixel1.emit(actualpixel1)
                        self.send_actualdifferent1.emit(actualdifferent1)
                        #self.send_contour1.emit(contour1)
                        #self.send_actuallengh1.emit(actuallengh1)

                        imgstack=cv2.cvtColor(imgstack,cv2.COLOR_BGR2RGB)
                        imgstack =QImage(imgstack,imgstack.shape[1],imgstack.shape[0],imgstack.strides[0],QImage.Format_RGB888)
                        #camera1 = pyqtSignal(QImage)
                        #anacamera1 = pyqtSignal(QImage)
                        self.anacamera1.emit(imgstack)                           
                self.img = cap.read()                
                alpha = 0.8
                mask_out = shapes_cap_out .astype(bool)
                self.img[mask_out] = cv2.addWeighted(self.cap_out,alpha,shapes_cap_out ,1-alpha,0)[mask_out]
                self.img =QImage(self.img,self.img.shape[1],self.img.shape[0],self.img.strides[0],QImage.Format_RGB888)
                self.camera1.emit(self.img)                          
            else:
                print("camera01 open failed")
class camera2(QThread):
    finished = pyqtSignal()
    camera2 = pyqtSignal(QImage)
    anacamera2 = pyqtSignal(QImage)
    send_actualpixel2 = pyqtSignal(int)
    send_actualdifferent2=pyqtSignal(int)
    send_contour2 =pyqtSignal(int)
    send_actuallengh2 =pyqtSignal(int)



    def __init__(self,read_database,pictureid,runmodel,qtycam2,save_picture_2):
        super().__init__()
        globalvar.read_database=read_database
        globalvar.pictureid=pictureid
        globalvar.runmodel=runmodel
        globalvar.qtycam2 = qtycam2
        globalvar.save_picture_2 = save_picture_2
        self.kernel = np.ones((3,3),np.uint8) 
        
    def stackImages(self,scale,imgArray):
      
        rows = len(imgArray)
        #print(rows)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0],list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0,rows):
                for y in range(0,cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)

                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height,width,3),np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0,rows):
                hor[x] =np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x],(0,0),None,scale,scale)
                    #imgArray[x] = cv2.resize(imgArray[x], (500, 400))
                else:
                    imgArray[x] = cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)
                    #imgArray[x] = cv2.resize(imgArray[x], (500, 400))
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)


            hor = np.hstack(imgArray)
            ver = hor
        return ver
    def load_data(self):
        #self.model_number = 642139
        
        connection = mysql.connector.connect(host='localhost',
                                            database = 'CAMERA_PAPER',
                                            user ='vqbg',
                                            password='thanhtam2408V!@')
        
        
        mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA_2 where Model = %s"
        val_select=(int(self.model_number),)
        cursor = connection.cursor()
        result_model = cursor.execute(mySql_Select_Table_Query,val_select)
        rows=cursor.fetchall()
        row_count=cursor.fetchmany()
        
        set_data = np.array(rows)
        cursor.close()
        connection.close()
        return set_data
    def find_nearest_points(self,parent_contour, child_contour):
        min_distance = float('inf')
        nearest_points = None

        # Lặp qua tất cả các điểm của contour cha
        for parent_point in parent_contour:
            x1, y1 = parent_point[0]

            # Lặp qua tất cả các điểm của contour con
            for child_point in child_contour:
                x2, y2 = child_point[0]

                # Tính khoảng cách giữa hai điểm
                distance = np.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)

                # Nếu khoảng cách nhỏ hơn khoảng cách hiện tại, cập nhật khoảng cách và tọa độ điểm
                if distance < min_distance:
                    min_distance = distance
                    nearest_points = [(x1, y1), (x2, y2)]

        return nearest_points, min_distance

    def run(self):
        #cap = nano.Camera(device_id=1,flip=0,width= 640,height= 480, fps= 30)
        #cap = nano.Camera(camera_type =1,device_id=0,width= 640,height= 480, fps= 30)

        diff_bit=0
        thres_bit=0
        center_ok=0
        M1=0
        M2=0

        while True:
            #set_data=self.load_data()
            #self.id_pic_check
            #print("array",set_data)
            M50_bit = client.read_coils(50,1)
            if M50_bit.isError():
                print("Error reading coil")
            else:
                #coil_value_Y0 = result_Y0.bits[0]  # Lấy trạng thái của coil tại địa chỉ 0
                #3print(f'Trạng thái của coil 00001: {coil_value_Y0}')
                M50 = M50_bit.bits[0]
            if M50==0:
                M1=1
            if M50==1 and M1== 1 and M2 ==0:
                #set_camera_control("exposure_auto",1)
                #set_camera_control("exposure_absolute",10)
                cap = nano.Camera(camera_type =1,device_id=0,width= 640,height= 480, fps= 30)
                #sleep(1)
            if M50==1 and cap.isReady():
                #sleep(5)
                M2 = M2 +1
                self.cap_out = cap.read()
                print("M50",M50)
                shapes_cap_out = np.zeros_like(self.cap_out,np.uint8)
                for r in range(globalvar.qtycam2):
                    cv2.rectangle(shapes_cap_out ,(globalvar.read_database[r,35],globalvar.read_database[r,33]),(globalvar.read_database[r,36],globalvar.read_database[r,34]),(255,0,0),4)
                    imgCropped= self.cap_out[globalvar.read_database[r,33]:globalvar.read_database[r,34],globalvar.read_database[r,35]:globalvar.read_database[r,36]]
                    imgCropped_1= self.cap_out[globalvar.read_database[r,33]:globalvar.read_database[r,34],globalvar.read_database[r,35]:globalvar.read_database[r,36]]
                    
                    alpha_b = (globalvar.read_database[r,58])/10
                    brightness_img = cv2.convertScaleAbs(imgCropped, alpha = alpha_b, beta = 0)
                    imgCropped = cv2.convertScaleAbs(imgCropped, alpha = alpha_b, beta = 0)
                    imgCropped_1 = cv2.convertScaleAbs(imgCropped_1, alpha = alpha_b, beta = 0)

                    contrast_factor = (globalvar.read_database[r,59])/10
                    contrasted_img = cv2.addWeighted(imgCropped,contrast_factor,np.zeros(imgCropped.shape,imgCropped.dtype),0,0)
                    imgCropped = cv2.addWeighted(imgCropped,contrast_factor,np.zeros(imgCropped.shape,imgCropped.dtype),0,0)
                    imgCropped_1 = cv2.addWeighted(imgCropped_1,contrast_factor,np.zeros(imgCropped_1.shape,imgCropped_1.dtype),0,0)
                    pixel = f"({imgCropped.shape[0]}, {imgCropped.shape[1]})"
                    cv2.putText(shapes_cap_out ,pixel, (globalvar.read_database[r,35],globalvar.read_database[r,34]), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),2)             
                    #imgCanny = cv2.Canny(imgCropped,50,100)

                    #imgHSV = cv2.cvtColor(imgCropped,cv2.COLOR_RGB2HSV)
                    #lower = np.array([set_data[r,25],set_data[r,27],set_data[r,29]])
                    #upper = np.array([set_data[r,26],set_data[r,28],set_data[r,30]])
                    #mask = cv2.inRange(imgHSV,lower,upper)
                    #imgResult= cv2.bitwise_and(imgCropped,imgCropped,mask=mask)
                    read_image = np.zeros_like(imgCropped,np.uint8)
                    diff = np.zeros_like(imgCropped,np.uint8)
                    diffBinary = np.zeros_like(imgCropped,np.uint8)
                    
                    #imgGray = cv2.cvtColor(imgCropped_1,cv2.COLOR_BGR2GRAY)
                    #imgBlur = cv2. GaussianBlur(imgGray,(3,3),0)
                    #self.read_database[r,10] : adjust threshold
                    #imgBinary= cv2.threshold(imgBlur,globalvar.read_database[r,39],255,cv2.THRESH_BINARY)[1]
                    #imgBinary = cv2.erode(imgBinary,self.kernel, iterations=globalvar.read_database[r,61])
                    #imgBinary = cv2.dilate(imgBinary,self.kernel, iterations=globalvar.read_database[r,60])
                    #actualpixel = cv2.countNonZero(imgBinary)


                    #read_image = imgCropped
                    #file_path = 'picture2_{}.jpg'
                    
                    #if os.path.exists(file_path.format(id_pic)):
                        #print("ton tai")
                    if r== 1:
                        thres_bit_mask=0
                        #imgCropped = cv2.add(imgCropped, set_data[r,36])
                        # alpha = 1.5  # hệ số tương phản
                        # beta = 50  # hệ số độ sáng
                        #imgCropped = cv2.convertScaleAbs(imgCropped, alpha=set_data[r, 34], beta=set_data[r, 35])
                        #mgGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
                        #imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
                        #imgBinary = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)[1]
                        #imgBinary_inv = cv2.bitwise_not(imgBinary)

                        height, width, channels = imgCropped.shape
                        ycenter= height // 2
                        xcenter = width // 2
                        mask_outside = np.zeros((height, width), dtype=np.uint8)
                        #mask_outside.fill(255)
                        mask_inside = np.zeros((height, width), dtype=np.uint8)
                        axis_major_outside, axis_minor_outside = int(globalvar.read_database[r,50]), int(globalvar.read_database[r,51])
                        axis_major_inside, axis_minor_inside = int(globalvar.read_database[r,53]), int(globalvar.read_database[r,54])

                        cv2.ellipse(mask_outside, (int(xcenter), int(ycenter)), (axis_major_outside, axis_minor_outside), globalvar.read_database[r,52], 0, 360, (255, 255, 255), -1)
                        cv2.ellipse(mask_inside, (int(xcenter), int(ycenter)),(axis_major_inside, axis_minor_inside), globalvar.read_database[r,55], 0, 360, (255, 255, 255),-1)

                        mask_check = cv2.bitwise_xor(mask_inside,mask_outside)
                        inside_inv = cv2.bitwise_not(mask_inside)
                        
                        inside= cv2.bitwise_and(imgCropped_1, imgCropped_1, mask=inside_inv)
                        outside = cv2.bitwise_and(inside, inside, mask=mask_outside)
                        outside_inv = cv2.bitwise_not(mask_outside)
                        
                        mask_all = cv2.bitwise_or(outside_inv,mask_inside)
                        imgCropped_1 = cv2.bitwise_not(outside, outside, mask=mask_all)

                        #imgCropped = cv2.bitwise_and(outside outside, mask=inside_inv)
                        #imgCropped = cv2.bitwise_not(imgCropped)
                        imgGray = cv2.cvtColor(imgCropped_1, cv2.COLOR_BGR2GRAY)
                        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
                        imgCropped_1 = imgBlur
                        imgBinary = cv2.threshold(imgBlur, globalvar.read_database[r,39], 255, cv2.THRESH_BINARY_INV)[1]
			
                        actualpixel = cv2.countNonZero(imgBinary)
                        
                        #imgBinary = cv2.erode(imgBinary,self.kernel, iterations=self.read_database[r,61])
                        #imgBinary = cv2.dilate(imgBinary,self.kernel, iterations=self.read_database[r,60])
                        #imgBinary_inv = cv2.bitwise_not(imgBinary)
                        #img_binary_bong= cv2.bitwise_and(imgCropped, imgCropped, mask=mask_check)
                        #imgGray_bong = cv2.cvtColor(img_binary_bong, cv2.COLOR_BGR2GRAY)
                        #imgBlur_bong = cv2.GaussianBlur(imgGray_bong, (3, 3), 0)
                        #imgBinary_bong = cv2.threshold(imgBlur_bong, set_data[r, 35], 255, cv2.THRESH_BINARY)[1]
                        #if thres_check>2000:
                            #imgBinary = cv2.bitwise_or(imgBinary_bong, imgBinary)
                            
                        read_image_1 = cv2.imread('picture2_{}.jpg'.format(r+1))

                        if imgCropped_1.shape != read_image_1.shape:
                            read_image_1 = cv2.resize(read_image_1,(imgCropped_1.shape[1],imgCropped_1.shape[0]))

                        gray2_1 = cv2.cvtColor(read_image_1, cv2.COLOR_BGR2GRAY)
                        diff_1 = cv2.absdiff(imgCropped_1, gray2_1)
                        diffBinary_1= cv2.threshold(diff_1,globalvar.read_database[r,42],255,cv2.THRESH_BINARY)[1]
                        diffBinary_1 = cv2.dilate(diffBinary_1,self.kernel,iterations=globalvar.read_database[r,60])

                        if center_ok ==1:
                            imgBinary = cv2.bitwise_or(imgBinary, diffBinary_1)
                            imgBinary = cv2.erode(imgBinary,self.kernel,iterations=globalvar.read_database[r,61])
                            imgBinary = cv2.dilate(imgBinary,self.kernel,iterations=globalvar.read_database[r,60])
                        #diff_bit = cv2.countNonZero(diffBinary)
                            actualpixel2 = cv2.countNonZero(imgBinary)
                        #thres_mask = cv2.countNonZero(mask_result)
                        len_cnt=0



                        contours, hierarchy = cv2.findContours(imgBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            len_cnt_in = int(cv2.arcLength(contours[0], True))
                            
                            if len_cnt_in > globalvar.read_database[r,48] and len_cnt_in < globalvar.read_database[r,49]:
                                actuallengh2 = int(cv2.arcLength(contours[0], True))
                                #len_cnt = cv2.arcLength(cnt)
                                #print(len_cnt)
                            #else:
                                #len_cnt = len_cnt_in

                        mask_cnt = np.zeros((height, width), dtype=np.uint8)
                        mask_cnt_1 = np.zeros((height, width), dtype=np.uint8)
                        
                        cv2.drawContours(mask_cnt, contours, -1, (255, 255, 255), -1)
                        cv2.drawContours(mask_cnt_1, contours, -1, (255, 255, 255), globalvar.read_database[r,56])
                        
                            
                        mask_cnt_out = cv2.bitwise_and(mask_cnt, mask_cnt, mask=mask_cnt_1)
                        imgBinary_cnt = cv2.bitwise_and(imgBinary, imgBinary, mask=mask_cnt_out)
                        imgBinary_diff = cv2.bitwise_not(imgBinary_cnt, imgBinary_cnt, mask=mask_cnt_out)                        
                        actualdifferent2 = cv2.countNonZero(imgBinary_diff)
                                             
                        read_image =mask_cnt
                        diff=mask_cnt_out
                        diffBinary=imgBinary_diff
                        cv2.drawContours(imgCropped, contours, -1, (0, 255, 0),globalvar.read_database[r,56])
                        cv2.drawContours(imgCropped, contours, -1, (0,0, 255), 2)
                        if actuallengh2 >globalvar.read_database[r,48] and actuallengh2<globalvar.read_database[r,49] and actualdifferent2 <= globalvar.read_database[r,44] :
                            GPIO.output(21, GPIO.HIGH)
                        else:
                            GPIO.output(21, GPIO.LOW)
        
                    elif r == 0:
                        read_image = cv2.imread('picture2_{}.jpg'.format(r+1))
                        if imgCropped.shape != read_image.shape:
                            read_image = cv2.resize(read_image,(imgCropped.shape[1],imgCropped.shape[0]))
                        gray1 = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
                        gray1 = cv2.GaussianBlur(gray1, (9, 9), 6)
                        gray2 = cv2.GaussianBlur(gray2, (9, 9), 6)
                        #diff = cv2.absdiff(gray1, gray2)
                        diffBinary1= cv2.threshold(gray1,globalvar.read_database[r,42],255,cv2.THRESH_BINARY)[1]
                        diffBinary2= cv2.threshold(gray2,globalvar.read_database[r,42],255,cv2.THRESH_BINARY)[1]
                        diffBinary1 = cv2.erode(diffBinary1,self.kernel, iterations=globalvar.read_database[r,61])
                        diffBinary1 = cv2.dilate(diffBinary1,self.kernel, iterations=globalvar.read_database[r,60])
                        diffBinary2 = cv2.erode(diffBinary2,self.kernel, iterations=globalvar.read_database[r,61])
                        diffBinary2 = cv2.dilate(diffBinary2,self.kernel, iterations=globalvar.read_database[r,60])
                        actualdifferent2 = cv2.countNonZero(diffBinary1)
                        actualpixel2 = cv2.countNonZero(diffBinary1)
                        actuallengh2 = cv2.countNonZero(diffBinary1)
                        Creal = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1= globalvar.read_database[r,48], param2= globalvar.read_database[r,49], minRadius=globalvar.read_database[r,50], maxRadius=globalvar.read_database[r,51])
                        Csample = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=globalvar.read_database[r,48], param2=globalvar.read_database[r,49], minRadius=globalvar.read_database[r,52], maxRadius=globalvar.read_database[r,53])
                        num_Creal = None
                        num_Csample = None
                        x_r = None
                        y_r = None
                        z_r = None
                        x_n = None
                        y_n = None
                        z_n = None
                        if Csample is not None :
                            num_Csample = Csample.shape[0]
                            #num_Creal = Creal.shape[0]
                            if num_Csample == 1 and client.connect(): 
                                Csample = np.round(Csample[0, :]).astype("int")
                                x_r,y_r,z_r = Csample[0]
                                cv2.circle(read_image,(x_r,y_r), 1, (0,255,255), 3)
                                cv2.circle(read_image, (x_r,y_r), z_r, (255,0,255), 3)
                                text = f"({x_r}, {y_r})"
                                text2 = f"({z_r})"
                                cv2.putText(read_image ,text, (x_r,y_r+20), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),2)
                                cv2.putText(read_image ,text2, (x_r,y_r+30), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),2)
                                #client.write_register(100, x_r)
                                #client.write_register(102, y_r)
                            else:
                                f=6
                        else:
                            #client.write_register(100, 1)
                            #client.write_register(102, 1)
                            f=5
                        if Creal is not None:
                            num_Creal = Creal.shape[0]
                            if M2 >3 and M1 ==1 and num_Creal == 1 and client.connect():
                                Creal = np.round(Creal[0, :]).astype("int")
                                x_n,y_n,z_n = Creal[0]
                                x_S = globalvar.read_database[r,35] + x_n
                                y_S = globalvar.read_database[r,33] + y_n
                                cv2.circle(imgCropped,(x_n,y_n), 1, (0,255,255), 3)
                                cv2.circle(imgCropped, (x_n,y_n), z_n, (255,0,255), 3)
                                text1 = f"({x_n}, {y_n})"
                                cv2.circle(shapes_cap_out, (x_S,y_S), z_n, (255,255,0), 3)
                                cv2.circle(shapes_cap_out, (x_S,y_S), 1, (255,255,0), 3)
                                cv2.putText(imgCropped ,text1, (x_n+10,y_n+10), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),2)
                                cv2.putText(shapes_cap_out ,text1, (x_S+10,y_S+10), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),2)
                                client.write_register(104, x_S)
                                client.write_register(106, y_S)
                                #if M2 ==50:
                                #cap.release()
                                M1=0
                                #M2 =0
                            else:
                                client.write_register(104, 0)
                                client.write_register(106, 0)
                                f=8
                        else:
                            client.write_register(104, 1)
                            client.write_register(106, 1)
                            #f=42
                        #if num_Creal == 1 and num_Csample == 1 and client.connect(): 
                                #x = x_n - x_r
                                #y = y_n - y_r
                                #client.write_register(100, x)
                                #client.write_register(102, y)
                                #client.write_register(104, x_r)
                                #client.write_register(106, y_r)
                                #client.write_register(108, x_n)
                                #client.write_register(110, y_n)
                                #print(f"{x},{y}")
                        #else:
                                #client.write_register(112, 100)
                                #client.write_register(114, 100)
                                #print("non")

                        #diff = cv2.absdiff(gray1, gray2)
                        #diffBinary= cv2.threshold(diff,globalvar.read_database[r,42],255,cv2.THRESH_BINARY)[1]
                        #diffBinary = cv2.erode(diffBinary,self.kernel, iterations=globalvar.read_database[r,61])
                        #diffBinary = cv2.dilate(diffBinary,self.kernel, iterations=globalvar.read_database[r,60])
                        #actualdifferent2 = cv2.countNonZero(diffBinary)
                        #actualpixel2 = cv2.countNonZero(diffBinary)
                        #actuallengh2 = cv2.countNonZero(diffBinary)
                        if (actualdifferent2> globalvar.read_database[r,44]) and client.connect():
                            client.write_coil(5, True)
                            GPIO.output(23,GPIO.HIGH)
                            cv2.putText(shapes_cap_out ,"ON",(globalvar.read_database[r,35],globalvar.read_database[r,33]),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,0),2)
                            center_ok =0
                        else:
                            GPIO.output(23,GPIO.LOW)
                            client.write_coil(5, False)
                            cv2.putText(shapes_cap_out ,"Off",(globalvar.read_database[r,35],globalvar.read_database[r,33]),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2)
                            center_ok =1
                                                    

                    else:
                        none=0
                    self.save_pic_ok= 1

                    # Địa chỉ IP của thiết bị
                    # Coils (đọc/ghi): Địa chỉ từ 00001 đến 09999 (bit). Y,M,T<C,...
                    # Discrete Inputs (chỉ đọc): Địa chỉ từ 10001 đến 19999 (bit). X
                    # Input Registers (chỉ đọc): Địa chỉ từ 30001 đến 39999 (16-bit).Only Read D0-->
                    # Holding Registers (đọc/ghi): Địa chỉ từ 40001 đến 49999 (16-bit). Read/WriteD0-->

                    if globalvar.read_database[r,1] == globalvar.pictureid:
                        filename = 'picture2_{}.jpg'
                        if r==1:
                            #send_actualpixel1 = pyqtSignal(int)
                            #send_actualdifferent1=pyqtSignal(int)
                            #send_contour1 =pyqtSignal(int)
                            #send_actuallengh1 =pyqtSignal(int)
                            self.send_actualpixel2.emit(actualpixel2)
                            self.send_actualdifferent2.emit(actualdifferent2)
                            #self.send_contour1.emit(contour1)
                            self.send_actuallengh2.emit(actuallengh2)
                            #self.contours_number.setValue(cnt_number)
                            imgstack = self.stackImages(1,([brightness_img,imgCropped, imgCropped_1, imgBinary], [contrasted_img,diff_1, diffBinary_1, diffBinary]))
                            imgstack=cv2.cvtColor(imgstack,cv2.COLOR_BGR2RGB)
                            imgstack =QImage(imgstack,imgstack.shape[1],imgstack.shape[0],imgstack.strides[0],QImage.Format_RGB888)
                            self.anacamera2.emit(imgstack)

                            if self.save_pic_ok == globalvar.save_picture_2:

                                cv2.imwrite(filename.format(globalvar.pictureid),imgBlur)
                        else:
                            if self.save_pic_ok == globalvar.save_picture_2:

                                cv2.imwrite(filename.format(globalvar.pictureid), imgCropped)

                            #send_actualpixel1 = pyqtSignal(int)
                            #send_actualdifferent1=pyqtSignal(int)
                            #send_contour1 =pyqtSignal(int)
                            #send_actuallengh1 =pyqtSignal(int)
                            self.send_actualpixel2.emit(actualpixel2)
                            self.send_actualdifferent2.emit(actualdifferent2)
                            #self.send_contour1.emit(contour1)
                            self.send_actuallengh2.emit(actuallengh2)
                            #self.contours_number.setValue(cnt_number)
                            imgstack = self.stackImages(1,([imgCropped, diffBinary1]))
                            imgstack=cv2.cvtColor(imgstack,cv2.COLOR_BGR2RGB)
                            imgstack =QImage(imgstack,imgstack.shape[1],imgstack.shape[0],imgstack.strides[0],QImage.Format_RGB888)
                            self.anacamera2.emit(imgstack)              

                self.img = cap.read()
                
                alpha = 0.8
                mask_out = shapes_cap_out .astype(bool)
                self.img[mask_out] = cv2.addWeighted(self.cap_out,alpha,shapes_cap_out ,1-alpha,0)[mask_out]
                self.img =QImage(self.img,self.img.shape[1],self.img.shape[0],self.img.strides[0],QImage.Format_RGB888)
                self.camera2.emit(self.img)
                
                #self.camera01.setPixmap(QtGui.QPixmap.fromImage(self.img))
                #sleep(5)
                if M1 ==0 or M2 ==30:
                    cap.release()
                    M1=0
                    M2 =0
                    client.write_register(100, 1)
                    client.write_register(102, 1)                            
            else:
                print("camera2 open failed")
                print("M50",M50)
            #if cv2.waitKey(1) &0xFF == ord('q'):
                #break   

class Ui_MainWindow(object):
    global start
        
    def __init__(self):
        super().__init__()
        #loader =QUiLoader()
        self.start=0
        self.read_database=[]
        self.save_picture_1 =0
        self.save_picture_2 =0
        self.pictureid=1
        self.runmodel= 642138
        self.qtycam1=1
        
        #start=0
        
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1352, 765)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_page = QtWidgets.QFrame(self.centralwidget)
        self.frame_page.setGeometry(QtCore.QRect(10, 50, 1331, 691))
        self.frame_page.setAutoFillBackground(False)
        self.frame_page.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_page.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_page.setObjectName("frame_page")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_page)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 1331, 691))
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.Box)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_0 = QtWidgets.QWidget()
        self.page_0.setObjectName("page_0")
        self.camera01frame = QtWidgets.QLabel(self.page_0)
        self.camera01frame.setGeometry(QtCore.QRect(170, 0, 1161, 691))
        self.camera01frame.setStyleSheet("background-color: rgb(252, 233, 79);")
        self.camera01frame.setAlignment(QtCore.Qt.AlignCenter)
        self.camera01frame.setObjectName("camera01frame")
        self.savedata1btn = QtWidgets.QPushButton(self.page_0)
        self.savedata1btn.setGeometry(QtCore.QRect(0, 580, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.savedata1btn.setFont(font)
        self.savedata1btn.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.savedata1btn.setObjectName("savedata1btn")
        self.X_Point_W_9 = QtWidgets.QLabel(self.page_0)
        self.X_Point_W_9.setGeometry(QtCore.QRect(0, 530, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_9.setFont(font)
        self.X_Point_W_9.setObjectName("X_Point_W_9")
        self.leftframe1box = QtWidgets.QSpinBox(self.page_0)
        self.leftframe1box.setGeometry(QtCore.QRect(20, 260, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.leftframe1box.setFont(font)
        self.leftframe1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.leftframe1box.setMaximum(999999)
        self.leftframe1box.setObjectName("leftframe1box")
        self.Y_Point_H_3 = QtWidgets.QLabel(self.page_0)
        self.Y_Point_H_3.setGeometry(QtCore.QRect(0, 230, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Y_Point_H_3.setFont(font)
        self.Y_Point_H_3.setObjectName("Y_Point_H_3")
        self.startcam1btn = QtWidgets.QPushButton(self.page_0)
        self.startcam1btn.setGeometry(QtCore.QRect(0, 70, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.startcam1btn.setFont(font)
        self.startcam1btn.setStyleSheet("background-color: rgb(193, 125, 17);")
        self.startcam1btn.setObjectName("startcam1btn")
        self.brightness1box = QtWidgets.QSpinBox(self.page_0)
        self.brightness1box.setGeometry(QtCore.QRect(20, 490, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.brightness1box.setFont(font)
        self.brightness1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.brightness1box.setMaximum(999999)
        self.brightness1box.setObjectName("brightness1box")
        self.rightframe1box = QtWidgets.QSpinBox(self.page_0)
        self.rightframe1box.setGeometry(QtCore.QRect(20, 330, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.rightframe1box.setFont(font)
        self.rightframe1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.rightframe1box.setMaximum(999999)
        self.rightframe1box.setObjectName("rightframe1box")
        self.X_Point_3 = QtWidgets.QLabel(self.page_0)
        self.X_Point_3.setGeometry(QtCore.QRect(0, 300, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_3.setFont(font)
        self.X_Point_3.setObjectName("X_Point_3")
        self.X_Point_W_10 = QtWidgets.QLabel(self.page_0)
        self.X_Point_W_10.setGeometry(QtCore.QRect(0, 450, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_10.setFont(font)
        self.X_Point_W_10.setObjectName("X_Point_W_10")
        self.savesample1btn = QtWidgets.QPushButton(self.page_0)
        self.savesample1btn.setGeometry(QtCore.QRect(0, 630, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.savesample1btn.setFont(font)
        self.savesample1btn.setStyleSheet("background-color: rgb(114, 159, 207);")
        self.savesample1btn.setObjectName("savesample1btn")
        self.contrast1box = QtWidgets.QSpinBox(self.page_0)
        self.contrast1box.setGeometry(QtCore.QRect(20, 410, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.contrast1box.setFont(font)
        self.contrast1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.contrast1box.setMaximum(999999)
        self.contrast1box.setObjectName("contrast1box")
        self.Y_Point_3 = QtWidgets.QLabel(self.page_0)
        self.Y_Point_3.setGeometry(QtCore.QRect(0, 160, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Y_Point_3.setFont(font)
        self.Y_Point_3.setObjectName("Y_Point_3")
        self.upperframe1box = QtWidgets.QSpinBox(self.page_0)
        self.upperframe1box.setGeometry(QtCore.QRect(20, 120, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.upperframe1box.setFont(font)
        self.upperframe1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.upperframe1box.setMaximum(999999)
        self.upperframe1box.setObjectName("upperframe1box")
        self.X_Point_W_3 = QtWidgets.QLabel(self.page_0)
        self.X_Point_W_3.setGeometry(QtCore.QRect(0, 380, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_3.setFont(font)
        self.X_Point_W_3.setObjectName("X_Point_W_3")
        self.lowerframe1box = QtWidgets.QSpinBox(self.page_0)
        self.lowerframe1box.setGeometry(QtCore.QRect(20, 190, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lowerframe1box.setFont(font)
        self.lowerframe1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.lowerframe1box.setMaximum(999999)
        self.lowerframe1box.setObjectName("lowerframe1box")
        self.stackedWidget.addWidget(self.page_0)
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName("page_1")
        self.Val_max_9 = QtWidgets.QLabel(self.page_1)
        self.Val_max_9.setGeometry(QtCore.QRect(1000, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_9.setFont(font)
        self.Val_max_9.setObjectName("Val_max_9")
        self.Sat_min_3 = QtWidgets.QLabel(self.page_1)
        self.Sat_min_3.setGeometry(QtCore.QRect(330, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sat_min_3.setFont(font)
        self.Sat_min_3.setObjectName("Sat_min_3")
        self.contour_number_7 = QtWidgets.QLabel(self.page_1)
        self.contour_number_7.setGeometry(QtCore.QRect(760, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_7.setFont(font)
        self.contour_number_7.setObjectName("contour_number_7")
        self.innerelipa1box = QtWidgets.QSpinBox(self.page_1)
        self.innerelipa1box.setGeometry(QtCore.QRect(330, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerelipa1box.setFont(font)
        self.innerelipa1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerelipa1box.setMaximum(999999)
        self.innerelipa1box.setObjectName("innerelipa1box")
        self.anacam1frame = QtWidgets.QLabel(self.page_1)
        self.anacam1frame.setGeometry(QtCore.QRect(0, 50, 1331, 591))
        self.anacam1frame.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.anacam1frame.setAlignment(QtCore.Qt.AlignCenter)
        self.anacam1frame.setObjectName("anacam1frame")
        self.Val_max_10 = QtWidgets.QLabel(self.page_1)
        self.Val_max_10.setGeometry(QtCore.QRect(550, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_10.setFont(font)
        self.Val_max_10.setObjectName("Val_max_10")
        self.Hue_min_3 = QtWidgets.QLabel(self.page_1)
        self.Hue_min_3.setGeometry(QtCore.QRect(0, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Hue_min_3.setFont(font)
        self.Hue_min_3.setObjectName("Hue_min_3")
        self.AreaOK_3 = QtWidgets.QLabel(self.page_1)
        self.AreaOK_3.setGeometry(QtCore.QRect(650, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.AreaOK_3.setFont(font)
        self.AreaOK_3.setObjectName("AreaOK_3")
        self.Val_max_11 = QtWidgets.QLabel(self.page_1)
        self.Val_max_11.setGeometry(QtCore.QRect(1110, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_11.setFont(font)
        self.Val_max_11.setObjectName("Val_max_11")
        self.Val_max_12 = QtWidgets.QLabel(self.page_1)
        self.Val_max_12.setGeometry(QtCore.QRect(660, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_12.setFont(font)
        self.Val_max_12.setObjectName("Val_max_12")
        self.setcontour1box = QtWidgets.QSpinBox(self.page_1)
        self.setcontour1box.setGeometry(QtCore.QRect(870, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setcontour1box.setFont(font)
        self.setcontour1box.setStyleSheet("background-color: rgb(252, 233, 79);")
        self.setcontour1box.setMaximum(999999)
        self.setcontour1box.setObjectName("setcontour1box")
        self.actualpixel1box = QtWidgets.QSpinBox(self.page_1)
        self.actualpixel1box.setGeometry(QtCore.QRect(200, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actualpixel1box.setFont(font)
        self.actualpixel1box.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.actualpixel1box.setMaximum(999999)
        self.actualpixel1box.setObjectName("actualpixel1box")
        self.numberconok_3 = QtWidgets.QLabel(self.page_1)
        self.numberconok_3.setGeometry(QtCore.QRect(430, 20, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.numberconok_3.setFont(font)
        self.numberconok_3.setObjectName("numberconok_3")
        self.innerelipb1box = QtWidgets.QSpinBox(self.page_1)
        self.innerelipb1box.setGeometry(QtCore.QRect(440, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerelipb1box.setFont(font)
        self.innerelipb1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerelipb1box.setMaximum(999999)
        self.innerelipb1box.setObjectName("innerelipb1box")
        self.outerdir1box = QtWidgets.QSpinBox(self.page_1)
        self.outerdir1box.setGeometry(QtCore.QRect(220, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerdir1box.setFont(font)
        self.outerdir1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerdir1box.setMaximum(999999)
        self.outerdir1box.setObjectName("outerdir1box")
        self.contour1box = QtWidgets.QSpinBox(self.page_1)
        self.contour1box.setGeometry(QtCore.QRect(760, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.contour1box.setFont(font)
        self.contour1box.setStyleSheet("background-color: rgb(252, 233, 79);")
        self.contour1box.setMaximum(999999)
        self.contour1box.setObjectName("contour1box")
        self.contour_number_8 = QtWidgets.QLabel(self.page_1)
        self.contour_number_8.setGeometry(QtCore.QRect(870, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_8.setFont(font)
        self.contour_number_8.setObjectName("contour_number_8")
        self.actuallengh1box = QtWidgets.QSpinBox(self.page_1)
        self.actuallengh1box.setGeometry(QtCore.QRect(980, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actuallengh1box.setFont(font)
        self.actuallengh1box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.actuallengh1box.setMaximum(999999)
        self.actuallengh1box.setObjectName("actuallengh1box")
        self.anacontrast1box = QtWidgets.QSpinBox(self.page_1)
        self.anacontrast1box.setGeometry(QtCore.QRect(1000, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.anacontrast1box.setFont(font)
        self.anacontrast1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.anacontrast1box.setMaximum(999999)
        self.anacontrast1box.setObjectName("anacontrast1box")
        self.outerelipa1box = QtWidgets.QSpinBox(self.page_1)
        self.outerelipa1box.setGeometry(QtCore.QRect(0, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerelipa1box.setFont(font)
        self.outerelipa1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerelipa1box.setMaximum(999999)
        self.outerelipa1box.setObjectName("outerelipa1box")
        self.kernelerode1box = QtWidgets.QSpinBox(self.page_1)
        self.kernelerode1box.setGeometry(QtCore.QRect(1220, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.kernelerode1box.setFont(font)
        self.kernelerode1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.kernelerode1box.setMaximum(999999)
        self.kernelerode1box.setObjectName("kernelerode1box")
        self.actualdifferent1box = QtWidgets.QSpinBox(self.page_1)
        self.actualdifferent1box.setGeometry(QtCore.QRect(540, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actualdifferent1box.setFont(font)
        self.actualdifferent1box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.actualdifferent1box.setMaximum(999999)
        self.actualdifferent1box.setObjectName("actualdifferent1box")
        self.contour_number_9 = QtWidgets.QLabel(self.page_1)
        self.contour_number_9.setGeometry(QtCore.QRect(1090, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_9.setFont(font)
        self.contour_number_9.setObjectName("contour_number_9")
        self.thresh_ok_3 = QtWidgets.QLabel(self.page_1)
        self.thresh_ok_3.setGeometry(QtCore.QRect(320, 20, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresh_ok_3.setFont(font)
        self.thresh_ok_3.setObjectName("thresh_ok_3")
        self.lenghmax1box = QtWidgets.QSpinBox(self.page_1)
        self.lenghmax1box.setGeometry(QtCore.QRect(1200, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lenghmax1box.setFont(font)
        self.lenghmax1box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.lenghmax1box.setMaximum(999999)
        self.lenghmax1box.setObjectName("lenghmax1box")
        self.kerneldilate1box = QtWidgets.QSpinBox(self.page_1)
        self.kerneldilate1box.setGeometry(QtCore.QRect(1110, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.kerneldilate1box.setFont(font)
        self.kerneldilate1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.kerneldilate1box.setMaximum(999999)
        self.kerneldilate1box.setObjectName("kerneldilate1box")
        self.innerdir1box = QtWidgets.QSpinBox(self.page_1)
        self.innerdir1box.setGeometry(QtCore.QRect(550, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerdir1box.setFont(font)
        self.innerdir1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerdir1box.setMaximum(999999)
        self.innerdir1box.setObjectName("innerdir1box")
        self.Val_min_3 = QtWidgets.QLabel(self.page_1)
        self.Val_min_3.setGeometry(QtCore.QRect(220, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_min_3.setFont(font)
        self.Val_min_3.setObjectName("Val_min_3")
        self.thresh_adj_3 = QtWidgets.QLabel(self.page_1)
        self.thresh_adj_3.setGeometry(QtCore.QRect(90, 20, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresh_adj_3.setFont(font)
        self.thresh_adj_3.setObjectName("thresh_adj_3")
        self.adjustdifferent1box = QtWidgets.QSpinBox(self.page_1)
        self.adjustdifferent1box.setGeometry(QtCore.QRect(430, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.adjustdifferent1box.setFont(font)
        self.adjustdifferent1box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.adjustdifferent1box.setMaximum(999999)
        self.adjustdifferent1box.setObjectName("adjustdifferent1box")
        self.threshold_3 = QtWidgets.QLabel(self.page_1)
        self.threshold_3.setGeometry(QtCore.QRect(200, 20, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshold_3.setFont(font)
        self.threshold_3.setObjectName("threshold_3")
        self.contour_number_10 = QtWidgets.QLabel(self.page_1)
        self.contour_number_10.setGeometry(QtCore.QRect(1200, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_10.setFont(font)
        self.contour_number_10.setObjectName("contour_number_10")
        self.setvalue1box = QtWidgets.QSpinBox(self.page_1)
        self.setvalue1box.setGeometry(QtCore.QRect(310, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setvalue1box.setFont(font)
        self.setvalue1box.setStyleSheet("\n"
"background-color: rgb(233, 185, 110);")
        self.setvalue1box.setMaximum(999999)
        self.setvalue1box.setObjectName("setvalue1box")
        self.contour_number_11 = QtWidgets.QLabel(self.page_1)
        self.contour_number_11.setGeometry(QtCore.QRect(980, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_11.setFont(font)
        self.contour_number_11.setObjectName("contour_number_11")
        self.outerelipb1box = QtWidgets.QSpinBox(self.page_1)
        self.outerelipb1box.setGeometry(QtCore.QRect(110, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerelipb1box.setFont(font)
        self.outerelipb1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerelipb1box.setMaximum(999999)
        self.outerelipb1box.setObjectName("outerelipb1box")
        self.adjustthreshold1box = QtWidgets.QSpinBox(self.page_1)
        self.adjustthreshold1box.setGeometry(QtCore.QRect(90, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.adjustthreshold1box.setFont(font)
        self.adjustthreshold1box.setStyleSheet("\n"
"background-color: rgb(233, 185, 110);")
        self.adjustthreshold1box.setMaximum(999999)
        self.adjustthreshold1box.setObjectName("adjustthreshold1box")
        self.setdifferent1box = QtWidgets.QSpinBox(self.page_1)
        self.setdifferent1box.setGeometry(QtCore.QRect(650, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setdifferent1box.setFont(font)
        self.setdifferent1box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.setdifferent1box.setMaximum(999999)
        self.setdifferent1box.setObjectName("setdifferent1box")
        self.Val_max_13 = QtWidgets.QLabel(self.page_1)
        self.Val_max_13.setGeometry(QtCore.QRect(1220, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_13.setFont(font)
        self.Val_max_13.setObjectName("Val_max_13")
        self.Sat_max_3 = QtWidgets.QLabel(self.page_1)
        self.Sat_max_3.setGeometry(QtCore.QRect(440, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sat_max_3.setFont(font)
        self.Sat_max_3.setObjectName("Sat_max_3")
        self.lenghmin1box = QtWidgets.QSpinBox(self.page_1)
        self.lenghmin1box.setGeometry(QtCore.QRect(1090, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lenghmin1box.setFont(font)
        self.lenghmin1box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.lenghmin1box.setMaximum(999999)
        self.lenghmin1box.setObjectName("lenghmin1box")
        self.Val_max_14 = QtWidgets.QLabel(self.page_1)
        self.Val_max_14.setGeometry(QtCore.QRect(890, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_14.setFont(font)
        self.Val_max_14.setObjectName("Val_max_14")
        self.contour_area_3 = QtWidgets.QLabel(self.page_1)
        self.contour_area_3.setGeometry(QtCore.QRect(530, 30, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_area_3.setFont(font)
        self.contour_area_3.setObjectName("contour_area_3")
        self.widthglue1box = QtWidgets.QSpinBox(self.page_1)
        self.widthglue1box.setGeometry(QtCore.QRect(660, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.widthglue1box.setFont(font)
        self.widthglue1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.widthglue1box.setMaximum(999999)
        self.widthglue1box.setObjectName("widthglue1box")
        self.anabrightness1box = QtWidgets.QSpinBox(self.page_1)
        self.anabrightness1box.setGeometry(QtCore.QRect(890, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.anabrightness1box.setFont(font)
        self.anabrightness1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.anabrightness1box.setMaximum(999999)
        self.anabrightness1box.setObjectName("anabrightness1box")
        self.Val_max_15 = QtWidgets.QLabel(self.page_1)
        self.Val_max_15.setGeometry(QtCore.QRect(760, 670, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_15.setFont(font)
        self.Val_max_15.setObjectName("Val_max_15")
        self.differentthreshold1box = QtWidgets.QSpinBox(self.page_1)
        self.differentthreshold1box.setGeometry(QtCore.QRect(770, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.differentthreshold1box.setFont(font)
        self.differentthreshold1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.differentthreshold1box.setMaximum(999999)
        self.differentthreshold1box.setObjectName("differentthreshold1box")
        self.Hue_max_3 = QtWidgets.QLabel(self.page_1)
        self.Hue_max_3.setGeometry(QtCore.QRect(110, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Hue_max_3.setFont(font)
        self.Hue_max_3.setObjectName("Hue_max_3")
        self.stackedWidget.addWidget(self.page_1)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.Y_Point_H_2 = QtWidgets.QLabel(self.page_2)
        self.Y_Point_H_2.setGeometry(QtCore.QRect(0, 210, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Y_Point_H_2.setFont(font)
        self.Y_Point_H_2.setObjectName("Y_Point_H_2")
        self.savedata2btn = QtWidgets.QPushButton(self.page_2)
        self.savedata2btn.setGeometry(QtCore.QRect(0, 560, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.savedata2btn.setFont(font)
        self.savedata2btn.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.savedata2btn.setObjectName("savedata2btn")
        self.Y_Point_2 = QtWidgets.QLabel(self.page_2)
        self.Y_Point_2.setGeometry(QtCore.QRect(0, 140, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Y_Point_2.setFont(font)
        self.Y_Point_2.setObjectName("Y_Point_2")
        self.lowerframe2box = QtWidgets.QSpinBox(self.page_2)
        self.lowerframe2box.setGeometry(QtCore.QRect(20, 170, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lowerframe2box.setFont(font)
        self.lowerframe2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.lowerframe2box.setMaximum(999999)
        self.lowerframe2box.setObjectName("lowerframe2box")
        self.X_Point_2 = QtWidgets.QLabel(self.page_2)
        self.X_Point_2.setGeometry(QtCore.QRect(0, 280, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_2.setFont(font)
        self.X_Point_2.setObjectName("X_Point_2")
        self.leftframe2box = QtWidgets.QSpinBox(self.page_2)
        self.leftframe2box.setGeometry(QtCore.QRect(20, 240, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.leftframe2box.setFont(font)
        self.leftframe2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.leftframe2box.setMaximum(999999)
        self.leftframe2box.setObjectName("leftframe2box")
        self.camera02frame = QtWidgets.QLabel(self.page_2)
        self.camera02frame.setGeometry(QtCore.QRect(180, 0, 1151, 691))
        self.camera02frame.setStyleSheet("background-color: rgb(195, 178, 150);")
        self.camera02frame.setAlignment(QtCore.Qt.AlignCenter)
        self.camera02frame.setObjectName("camera02frame")
        self.upperframe2box = QtWidgets.QSpinBox(self.page_2)
        self.upperframe2box.setGeometry(QtCore.QRect(20, 100, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.upperframe2box.setFont(font)
        self.upperframe2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.upperframe2box.setMaximum(999999)
        self.upperframe2box.setObjectName("upperframe2box")
        self.rightframe2box = QtWidgets.QSpinBox(self.page_2)
        self.rightframe2box.setGeometry(QtCore.QRect(20, 310, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.rightframe2box.setFont(font)
        self.rightframe2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.rightframe2box.setMaximum(999999)
        self.rightframe2box.setObjectName("rightframe2box")
        self.X_Point_W_2 = QtWidgets.QLabel(self.page_2)
        self.X_Point_W_2.setGeometry(QtCore.QRect(0, 360, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_2.setFont(font)
        self.X_Point_W_2.setObjectName("X_Point_W_2")
        self.contrast2box = QtWidgets.QSpinBox(self.page_2)
        self.contrast2box.setGeometry(QtCore.QRect(20, 390, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.contrast2box.setFont(font)
        self.contrast2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.contrast2box.setMaximum(999999)
        self.contrast2box.setObjectName("contrast2box")
        self.X_Point_W_8 = QtWidgets.QLabel(self.page_2)
        self.X_Point_W_8.setGeometry(QtCore.QRect(0, 430, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_8.setFont(font)
        self.X_Point_W_8.setObjectName("X_Point_W_8")
        self.brightness2box = QtWidgets.QSpinBox(self.page_2)
        self.brightness2box.setGeometry(QtCore.QRect(20, 470, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.brightness2box.setFont(font)
        self.brightness2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.brightness2box.setMaximum(999999)
        self.brightness2box.setObjectName("brightness2box")
        self.X_Point_W_7 = QtWidgets.QLabel(self.page_2)
        self.X_Point_W_7.setGeometry(QtCore.QRect(0, 510, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.X_Point_W_7.setFont(font)
        self.X_Point_W_7.setObjectName("X_Point_W_7")
        self.startcam2btn = QtWidgets.QPushButton(self.page_2)
        self.startcam2btn.setGeometry(QtCore.QRect(0, 50, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.startcam2btn.setFont(font)
        self.startcam2btn.setStyleSheet("background-color: rgb(193, 125, 17);")
        self.startcam2btn.setObjectName("startcam2btn")
        self.savesample2btn = QtWidgets.QPushButton(self.page_2)
        self.savesample2btn.setGeometry(QtCore.QRect(0, 610, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.savesample2btn.setFont(font)
        self.savesample2btn.setStyleSheet("background-color: rgb(114, 159, 207);")
        self.savesample2btn.setObjectName("savesample2btn")
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.actualpixel2box = QtWidgets.QSpinBox(self.page_3)
        self.actualpixel2box.setGeometry(QtCore.QRect(200, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actualpixel2box.setFont(font)
        self.actualpixel2box.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.actualpixel2box.setMaximum(999999)
        self.actualpixel2box.setObjectName("actualpixel2box")
        self.thresh_ok_2 = QtWidgets.QLabel(self.page_3)
        self.thresh_ok_2.setGeometry(QtCore.QRect(320, 20, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresh_ok_2.setFont(font)
        self.thresh_ok_2.setObjectName("thresh_ok_2")
        self.outerelipb2box = QtWidgets.QSpinBox(self.page_3)
        self.outerelipb2box.setGeometry(QtCore.QRect(110, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerelipb2box.setFont(font)
        self.outerelipb2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerelipb2box.setMaximum(999999)
        self.outerelipb2box.setObjectName("outerelipb2box")
        self.setvalue2box = QtWidgets.QSpinBox(self.page_3)
        self.setvalue2box.setGeometry(QtCore.QRect(310, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setvalue2box.setFont(font)
        self.setvalue2box.setStyleSheet("\n"
"background-color: rgb(233, 185, 110);")
        self.setvalue2box.setMaximum(999999)
        self.setvalue2box.setObjectName("setvalue2box")
        self.numberconok_2 = QtWidgets.QLabel(self.page_3)
        self.numberconok_2.setGeometry(QtCore.QRect(430, 20, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.numberconok_2.setFont(font)
        self.numberconok_2.setObjectName("numberconok_2")
        self.AreaOK_2 = QtWidgets.QLabel(self.page_3)
        self.AreaOK_2.setGeometry(QtCore.QRect(650, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.AreaOK_2.setFont(font)
        self.AreaOK_2.setObjectName("AreaOK_2")
        self.Sat_max_2 = QtWidgets.QLabel(self.page_3)
        self.Sat_max_2.setGeometry(QtCore.QRect(440, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sat_max_2.setFont(font)
        self.Sat_max_2.setObjectName("Sat_max_2")
        self.Hue_min_2 = QtWidgets.QLabel(self.page_3)
        self.Hue_min_2.setGeometry(QtCore.QRect(0, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Hue_min_2.setFont(font)
        self.Hue_min_2.setObjectName("Hue_min_2")
        self.innerelipa2box = QtWidgets.QSpinBox(self.page_3)
        self.innerelipa2box.setGeometry(QtCore.QRect(330, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerelipa2box.setFont(font)
        self.innerelipa2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerelipa2box.setMaximum(999999)
        self.innerelipa2box.setObjectName("innerelipa2box")
        self.setdifferent2box = QtWidgets.QSpinBox(self.page_3)
        self.setdifferent2box.setGeometry(QtCore.QRect(650, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setdifferent2box.setFont(font)
        self.setdifferent2box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.setdifferent2box.setMaximum(999999)
        self.setdifferent2box.setObjectName("setdifferent2box")
        self.adjustthreshold2box = QtWidgets.QSpinBox(self.page_3)
        self.adjustthreshold2box.setGeometry(QtCore.QRect(90, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.adjustthreshold2box.setFont(font)
        self.adjustthreshold2box.setStyleSheet("\n"
"background-color: rgb(233, 185, 110);")
        self.adjustthreshold2box.setMaximum(999999)
        self.adjustthreshold2box.setObjectName("adjustthreshold2box")
        self.outerdir2box = QtWidgets.QSpinBox(self.page_3)
        self.outerdir2box.setGeometry(QtCore.QRect(220, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerdir2box.setFont(font)
        self.outerdir2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerdir2box.setMaximum(999999)
        self.outerdir2box.setObjectName("outerdir2box")
        self.innerelipb2box = QtWidgets.QSpinBox(self.page_3)
        self.innerelipb2box.setGeometry(QtCore.QRect(440, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerelipb2box.setFont(font)
        self.innerelipb2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerelipb2box.setMaximum(999999)
        self.innerelipb2box.setObjectName("innerelipb2box")
        self.Val_min_2 = QtWidgets.QLabel(self.page_3)
        self.Val_min_2.setGeometry(QtCore.QRect(220, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_min_2.setFont(font)
        self.Val_min_2.setObjectName("Val_min_2")
        self.Val_max_2 = QtWidgets.QLabel(self.page_3)
        self.Val_max_2.setGeometry(QtCore.QRect(550, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_2.setFont(font)
        self.Val_max_2.setObjectName("Val_max_2")
        self.Hue_max_2 = QtWidgets.QLabel(self.page_3)
        self.Hue_max_2.setGeometry(QtCore.QRect(110, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Hue_max_2.setFont(font)
        self.Hue_max_2.setObjectName("Hue_max_2")
        self.contour_area_2 = QtWidgets.QLabel(self.page_3)
        self.contour_area_2.setGeometry(QtCore.QRect(530, 30, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_area_2.setFont(font)
        self.contour_area_2.setObjectName("contour_area_2")
        self.anacam2frame = QtWidgets.QLabel(self.page_3)
        self.anacam2frame.setGeometry(QtCore.QRect(10, 50, 1321, 591))
        self.anacam2frame.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.anacam2frame.setAlignment(QtCore.Qt.AlignCenter)
        self.anacam2frame.setObjectName("anacam2frame")
        self.contour2box = QtWidgets.QSpinBox(self.page_3)
        self.contour2box.setGeometry(QtCore.QRect(760, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.contour2box.setFont(font)
        self.contour2box.setStyleSheet("background-color: rgb(252, 233, 79);")
        self.contour2box.setMaximum(999999)
        self.contour2box.setObjectName("contour2box")
        self.threshold_2 = QtWidgets.QLabel(self.page_3)
        self.threshold_2.setGeometry(QtCore.QRect(200, 20, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshold_2.setFont(font)
        self.threshold_2.setObjectName("threshold_2")
        self.contour_number_2 = QtWidgets.QLabel(self.page_3)
        self.contour_number_2.setGeometry(QtCore.QRect(760, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_2.setFont(font)
        self.contour_number_2.setObjectName("contour_number_2")
        self.innerdir2box = QtWidgets.QSpinBox(self.page_3)
        self.innerdir2box.setGeometry(QtCore.QRect(550, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.innerdir2box.setFont(font)
        self.innerdir2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.innerdir2box.setMaximum(999999)
        self.innerdir2box.setObjectName("innerdir2box")
        self.thresh_adj_2 = QtWidgets.QLabel(self.page_3)
        self.thresh_adj_2.setGeometry(QtCore.QRect(90, 20, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresh_adj_2.setFont(font)
        self.thresh_adj_2.setObjectName("thresh_adj_2")
        self.adjustdifferent2box = QtWidgets.QSpinBox(self.page_3)
        self.adjustdifferent2box.setGeometry(QtCore.QRect(430, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.adjustdifferent2box.setFont(font)
        self.adjustdifferent2box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.adjustdifferent2box.setMaximum(999999)
        self.adjustdifferent2box.setObjectName("adjustdifferent2box")
        self.actualdifferent2box = QtWidgets.QSpinBox(self.page_3)
        self.actualdifferent2box.setGeometry(QtCore.QRect(540, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actualdifferent2box.setFont(font)
        self.actualdifferent2box.setStyleSheet("background-color: rgb(173, 127, 168);")
        self.actualdifferent2box.setMaximum(999999)
        self.actualdifferent2box.setObjectName("actualdifferent2box")
        self.outerelipa2box = QtWidgets.QSpinBox(self.page_3)
        self.outerelipa2box.setGeometry(QtCore.QRect(0, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.outerelipa2box.setFont(font)
        self.outerelipa2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.outerelipa2box.setMaximum(999999)
        self.outerelipa2box.setObjectName("outerelipa2box")
        self.Sat_min_2 = QtWidgets.QLabel(self.page_3)
        self.Sat_min_2.setGeometry(QtCore.QRect(330, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sat_min_2.setFont(font)
        self.Sat_min_2.setObjectName("Sat_min_2")
        self.setcontour2box = QtWidgets.QSpinBox(self.page_3)
        self.setcontour2box.setGeometry(QtCore.QRect(870, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setcontour2box.setFont(font)
        self.setcontour2box.setStyleSheet("background-color: rgb(252, 233, 79);")
        self.setcontour2box.setMaximum(999999)
        self.setcontour2box.setObjectName("setcontour2box")
        self.contour_number_3 = QtWidgets.QLabel(self.page_3)
        self.contour_number_3.setGeometry(QtCore.QRect(870, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_3.setFont(font)
        self.contour_number_3.setObjectName("contour_number_3")
        self.actuallengh2box = QtWidgets.QSpinBox(self.page_3)
        self.actuallengh2box.setGeometry(QtCore.QRect(980, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.actuallengh2box.setFont(font)
        self.actuallengh2box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.actuallengh2box.setMaximum(999999)
        self.actuallengh2box.setObjectName("actuallengh2box")
        self.contour_number_4 = QtWidgets.QLabel(self.page_3)
        self.contour_number_4.setGeometry(QtCore.QRect(980, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_4.setFont(font)
        self.contour_number_4.setObjectName("contour_number_4")
        self.lenghmin2box = QtWidgets.QSpinBox(self.page_3)
        self.lenghmin2box.setGeometry(QtCore.QRect(1090, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lenghmin2box.setFont(font)
        self.lenghmin2box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.lenghmin2box.setMaximum(999999)
        self.lenghmin2box.setObjectName("lenghmin2box")
        self.contour_number_5 = QtWidgets.QLabel(self.page_3)
        self.contour_number_5.setGeometry(QtCore.QRect(1090, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_5.setFont(font)
        self.contour_number_5.setObjectName("contour_number_5")
        self.lenghmax2box = QtWidgets.QSpinBox(self.page_3)
        self.lenghmax2box.setGeometry(QtCore.QRect(1200, 0, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lenghmax2box.setFont(font)
        self.lenghmax2box.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.lenghmax2box.setMaximum(999999)
        self.lenghmax2box.setObjectName("lenghmax2box")
        self.contour_number_6 = QtWidgets.QLabel(self.page_3)
        self.contour_number_6.setGeometry(QtCore.QRect(1200, 30, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_number_6.setFont(font)
        self.contour_number_6.setObjectName("contour_number_6")
        self.Val_max_3 = QtWidgets.QLabel(self.page_3)
        self.Val_max_3.setGeometry(QtCore.QRect(890, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_3.setFont(font)
        self.Val_max_3.setObjectName("Val_max_3")
        self.anabrightness2box = QtWidgets.QSpinBox(self.page_3)
        self.anabrightness2box.setGeometry(QtCore.QRect(890, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.anabrightness2box.setFont(font)
        self.anabrightness2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.anabrightness2box.setMaximum(999999)
        self.anabrightness2box.setObjectName("anabrightness2box")
        self.Val_max_4 = QtWidgets.QLabel(self.page_3)
        self.Val_max_4.setGeometry(QtCore.QRect(1000, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_4.setFont(font)
        self.Val_max_4.setObjectName("Val_max_4")
        self.anacontrast2box = QtWidgets.QSpinBox(self.page_3)
        self.anacontrast2box.setGeometry(QtCore.QRect(1000, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.anacontrast2box.setFont(font)
        self.anacontrast2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.anacontrast2box.setMaximum(999999)
        self.anacontrast2box.setObjectName("anacontrast2box")
        self.kerneldilate2box = QtWidgets.QSpinBox(self.page_3)
        self.kerneldilate2box.setGeometry(QtCore.QRect(1110, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.kerneldilate2box.setFont(font)
        self.kerneldilate2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.kerneldilate2box.setMaximum(999999)
        self.kerneldilate2box.setObjectName("kerneldilate2box")
        self.Val_max_5 = QtWidgets.QLabel(self.page_3)
        self.Val_max_5.setGeometry(QtCore.QRect(1110, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_5.setFont(font)
        self.Val_max_5.setObjectName("Val_max_5")
        self.Val_max_6 = QtWidgets.QLabel(self.page_3)
        self.Val_max_6.setGeometry(QtCore.QRect(660, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_6.setFont(font)
        self.Val_max_6.setObjectName("Val_max_6")
        self.widthglue2box = QtWidgets.QSpinBox(self.page_3)
        self.widthglue2box.setGeometry(QtCore.QRect(660, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.widthglue2box.setFont(font)
        self.widthglue2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.widthglue2box.setMaximum(999999)
        self.widthglue2box.setObjectName("widthglue2box")
        self.Val_max_7 = QtWidgets.QLabel(self.page_3)
        self.Val_max_7.setGeometry(QtCore.QRect(760, 670, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_7.setFont(font)
        self.Val_max_7.setObjectName("Val_max_7")
        self.differentthreshold2box = QtWidgets.QSpinBox(self.page_3)
        self.differentthreshold2box.setGeometry(QtCore.QRect(770, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.differentthreshold2box.setFont(font)
        self.differentthreshold2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.differentthreshold2box.setMaximum(999999)
        self.differentthreshold2box.setObjectName("differentthreshold2box")
        self.kernelerode2box = QtWidgets.QSpinBox(self.page_3)
        self.kernelerode2box.setGeometry(QtCore.QRect(1220, 640, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.kernelerode2box.setFont(font)
        self.kernelerode2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.kernelerode2box.setMaximum(999999)
        self.kernelerode2box.setObjectName("kernelerode2box")
        self.Val_max_8 = QtWidgets.QLabel(self.page_3)
        self.Val_max_8.setGeometry(QtCore.QRect(1220, 670, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Val_max_8.setFont(font)
        self.Val_max_8.setObjectName("Val_max_8")
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.stackedWidget.addWidget(self.page_4)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.stackedWidget.addWidget(self.page_6)
        self.page_7 = QtWidgets.QWidget()
        self.page_7.setObjectName("page_7")
        self.stackedWidget.addWidget(self.page_7)
        self.page_8 = QtWidgets.QWidget()
        self.page_8.setObjectName("page_8")
        self.stackedWidget.addWidget(self.page_8)
        self.page_9 = QtWidgets.QWidget()
        self.page_9.setObjectName("page_9")
        self.stackedWidget.addWidget(self.page_9)
        self.page_10 = QtWidgets.QWidget()
        self.page_10.setObjectName("page_10")
        self.stackedWidget.addWidget(self.page_10)
        self.video1btn = QtWidgets.QPushButton(self.centralwidget)
        self.video1btn.setGeometry(QtCore.QRect(0, 0, 81, 41))
        self.video1btn.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.video1btn.setObjectName("video1btn")
        self.anavideo2btn = QtWidgets.QPushButton(self.centralwidget)
        self.anavideo2btn.setGeometry(QtCore.QRect(330, 0, 131, 41))
        self.anavideo2btn.setStyleSheet("background-color: rgb(52, 101, 164);")
        self.anavideo2btn.setObjectName("anavideo2btn")
        self.video2btn = QtWidgets.QPushButton(self.centralwidget)
        self.video2btn.setGeometry(QtCore.QRect(220, 0, 101, 41))
        self.video2btn.setStyleSheet("background-color: rgb(114, 159, 207);")
        self.video2btn.setObjectName("video2btn")
        self.anavideo1btn = QtWidgets.QPushButton(self.centralwidget)
        self.anavideo1btn.setGeometry(QtCore.QRect(90, 0, 111, 41))
        self.anavideo1btn.setStyleSheet("background-color: rgb(245, 121, 0);")
        self.anavideo1btn.setObjectName("anavideo1btn")
        self.runmodelbox = QtWidgets.QSpinBox(self.centralwidget)
        self.runmodelbox.setGeometry(QtCore.QRect(700, 0, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.runmodelbox.setFont(font)
        self.runmodelbox.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.runmodelbox.setMaximum(999999)
        self.runmodelbox.setObjectName("runmodelbox")
        self.Model = QtWidgets.QLabel(self.centralwidget)
        self.Model.setGeometry(QtCore.QRect(700, 30, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Model.setFont(font)
        self.Model.setObjectName("Model")
        self.createdatabtn = QtWidgets.QPushButton(self.centralwidget)
        self.createdatabtn.setGeometry(QtCore.QRect(470, 0, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.createdatabtn.setFont(font)
        self.createdatabtn.setStyleSheet("background-color: rgb(92, 53, 102);")
        self.createdatabtn.setObjectName("createdatabtn")
        self.Model_3 = QtWidgets.QLabel(self.centralwidget)
        self.Model_3.setGeometry(QtCore.QRect(1100, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Model_3.setFont(font)
        self.Model_3.setObjectName("Model_3")
        self.qtycam1box = QtWidgets.QSpinBox(self.centralwidget)
        self.qtycam1box.setGeometry(QtCore.QRect(1100, 0, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.qtycam1box.setFont(font)
        self.qtycam1box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.qtycam1box.setMaximum(999999)
        self.qtycam1box.setObjectName("qtycam1box")
        self.Model_4 = QtWidgets.QLabel(self.centralwidget)
        self.Model_4.setGeometry(QtCore.QRect(1190, 30, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Model_4.setFont(font)
        self.Model_4.setObjectName("Model_4")
        self.qtycam2box = QtWidgets.QSpinBox(self.centralwidget)
        self.qtycam2box.setGeometry(QtCore.QRect(1190, 0, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.qtycam2box.setFont(font)
        self.qtycam2box.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.qtycam2box.setMaximum(999999)
        self.qtycam2box.setObjectName("qtycam2box")
        self.Model_5 = QtWidgets.QLabel(self.centralwidget)
        self.Model_5.setGeometry(QtCore.QRect(940, 30, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Model_5.setFont(font)
        self.Model_5.setObjectName("Model_5")
        self.copymodelbox = QtWidgets.QSpinBox(self.centralwidget)
        self.copymodelbox.setGeometry(QtCore.QRect(940, 0, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.copymodelbox.setFont(font)
        self.copymodelbox.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.copymodelbox.setMaximum(999999)
        self.copymodelbox.setObjectName("copymodelbox")
        self.copydatabtn = QtWidgets.QPushButton(self.centralwidget)
        self.copydatabtn.setGeometry(QtCore.QRect(840, 0, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.copydatabtn.setFont(font)
        self.copydatabtn.setStyleSheet("background-color: rgb(252, 175, 62);")
        self.copydatabtn.setObjectName("copydatabtn")
        self.Model_2 = QtWidgets.QLabel(self.centralwidget)
        self.Model_2.setGeometry(QtCore.QRect(20, 80, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Model_2.setFont(font)
        self.Model_2.setObjectName("Model_2")
        self.pictureidbox = QtWidgets.QSpinBox(self.centralwidget)
        self.pictureidbox.setGeometry(QtCore.QRect(10, 50, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pictureidbox.setFont(font)
        self.pictureidbox.setStyleSheet("background-color: rgb(138, 226, 52);\n"
"")
        self.pictureidbox.setMaximum(999999)
        self.pictureidbox.setObjectName("pictureidbox")
        self.deletedatabtn = QtWidgets.QPushButton(self.centralwidget)
        self.deletedatabtn.setGeometry(QtCore.QRect(590, 0, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.deletedatabtn.setFont(font)
        self.deletedatabtn.setStyleSheet("background-color: rgb(239, 41, 41);")
        self.deletedatabtn.setObjectName("deletedatabtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)
        # chuyen cac trang
        self.video1btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.anavideo1btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.video2btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.anavideo2btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        #Xu ly cac  button va box main
        self.createdatabtn.clicked.connect(self.createdatabtn_fun)
        self.deletedatabtn.clicked.connect(self.deletedatabtn_fun)
        self.runmodelbox.valueChanged['int'].connect(self.runmodelbox_fun)
        self.copydatabtn.clicked.connect(self.copydatabtn_fun)
        self.copymodelbox.valueChanged['int'].connect(self.copymodelbox_fun)
        self.qtycam1box.valueChanged['int'].connect(self.qtycam1box_fun)
        self.qtycam2box.valueChanged['int'].connect(self.qtycam2box_fun)
        self.pictureidbox.valueChanged['int'].connect(self.pictureidbox_fun)
        # xu ly da ta video 1
        self.startcam1btn.clicked.connect(self.startcam1btn_fun)
        self.upperframe1box.valueChanged['int'].connect(self.upperframe1box_fun)
        self.lowerframe1box.valueChanged['int'].connect(self.lowerframe1box_fun)
        self.leftframe1box.valueChanged['int'].connect(self.leftframe1box_fun)
        self.rightframe1box.valueChanged['int'].connect(self.rightframe1box_fun)
        self.contrast1box.valueChanged['int'].connect(self.contrast1box_fun)
        self.brightness1box.valueChanged['int'].connect(self.brightness1box_fun)
        self.savedata1btn.clicked.connect(self.savedata1btn_fun)
        self.savesample1btn.clicked.connect(self.savesample1btn_fun)
        #xu ly da ta phan tich video 1
        self.adjustthreshold1box.valueChanged['int'].connect(self.adjustthreshold1box_fun)
        self.actualpixel1box.valueChanged['int'].connect(self.actualpixel1box_fun)
        self.setvalue1box.valueChanged['int'].connect(self.setvalue1box_fun)
        self.adjustdifferent1box.valueChanged['int'].connect(self.adjustdifferent1box_fun)
        self.actualdifferent1box.valueChanged['int'].connect(self.actualdifferent1box_fun)
        self.setdifferent1box.valueChanged['int'].connect(self.setdifferent1box_fun)
        self.contour1box.valueChanged['int'].connect(self.contour1box_fun)
        self.setcontour1box.valueChanged['int'].connect(self.setcontour1box_fun)
        self.actuallengh1box.valueChanged['int'].connect(self.actuallengh1box_fun)
        self.lenghmin1box.valueChanged['int'].connect(self.lenghmin1box_fun)
        self.lenghmax1box.valueChanged['int'].connect(self.lenghmax1box_fun)
        self.outerelipa1box.valueChanged['int'].connect(self.outerelipa1box_fun)
        self.outerelipb1box.valueChanged['int'].connect(self.outerelipb1box_fun)
        self.outerdir1box.valueChanged['int'].connect(self.outerdir1box_fun)
        self.innerelipa1box.valueChanged['int'].connect(self.innerelipa1box_fun)
        self.innerelipb1box.valueChanged['int'].connect(self.innerelipb1box_fun)
        self.innerdir1box.valueChanged['int'].connect(self.innerdir1box_fun)
        self.widthglue1box.valueChanged['int'].connect(self.widthglue1box_fun)
        self.differentthreshold1box.valueChanged['int'].connect(self.differentthreshold1box_fun)
        self.anabrightness1box.valueChanged['int'].connect(self.anabrightness1box_fun)
        self.anacontrast1box.valueChanged['int'].connect(self.anacontrast1box_fun)
        self.kerneldilate1box.valueChanged['int'].connect(self.kerneldilate1box_fun)
        self.kernelerode1box.valueChanged['int'].connect(self.kernelerode1box_fun)
        #xu ly du lieu video 2
        self.startcam2btn.clicked.connect(self.startcam2btn_fun)
        self.upperframe2box.valueChanged['int'].connect(self.upperframe2box_fun)
        self.lowerframe2box.valueChanged['int'].connect(self.lowerframe2box_fun)
        self.leftframe2box.valueChanged['int'].connect(self.leftframe2box_fun)
        self.rightframe2box.valueChanged['int'].connect(self.rightframe2box_fun)
        self.contrast2box.valueChanged['int'].connect(self.contrast2box_fun)
        self.brightness2box.valueChanged['int'].connect(self.brightness2box_fun)
        self.savedata2btn.clicked.connect(self.savedata1btn_fun)
        self.savesample2btn.clicked.connect(self.savesample2btn_fun)
        #phan tich du lieu video 2
        self.adjustthreshold2box.valueChanged['int'].connect(self.adjustthreshold2box_fun)
        self.actualpixel2box.valueChanged['int'].connect(self.actualpixel2box_fun)
        self.setvalue2box.valueChanged['int'].connect(self.setvalue2box_fun)
        self.adjustdifferent2box.valueChanged['int'].connect(self.adjustdifferent2box_fun)
        self.actualdifferent2box.valueChanged['int'].connect(self.actualdifferent2box_fun)
        self.setdifferent2box.valueChanged['int'].connect(self.setdifferent2box_fun)
        self.contour2box.valueChanged['int'].connect(self.contour2box_fun)
        self.setcontour2box.valueChanged['int'].connect(self.setcontour2box_fun)
        self.actuallengh2box.valueChanged['int'].connect(self.actuallengh2box_fun)
        self.lenghmin2box.valueChanged['int'].connect(self.lenghmin2box_fun)
        self.lenghmax2box.valueChanged['int'].connect(self.lenghmax2box_fun)
        self.outerelipa2box.valueChanged['int'].connect(self.outerelipa2box_fun)
        self.outerelipb2box.valueChanged['int'].connect(self.outerelipb2box_fun)
        self.outerdir2box.valueChanged['int'].connect(self.outerdir2box_fun)
        self.innerelipa2box.valueChanged['int'].connect(self.innerelipa2box_fun)
        self.innerelipb2box.valueChanged['int'].connect(self.innerelipb2box_fun)
        self.innerdir2box.valueChanged['int'].connect(self.innerdir2box_fun)
        self.widthglue2box.valueChanged['int'].connect(self.widthglue2box_fun)
        self.differentthreshold2box.valueChanged['int'].connect(self.differentthreshold2box_fun)
        self.anabrightness2box.valueChanged['int'].connect(self.anabrightness2box_fun)
        self.anacontrast2box.valueChanged['int'].connect(self.anacontrast2box_fun)
        self.kerneldilate2box.valueChanged['int'].connect(self.kerneldilate2box_fun)
        self.kernelerode2box.valueChanged['int'].connect(self.kernelerode2box_fun)
        self.loaddata_start()
        
        # xu ly ham lay du lieu va luu du lieu cho chuong trinh
    def show_dialog1(self):
        dialog = QMessageBox(MainWindow)
        dialog.setText(' Model da ton tai')
        dialog.setIcon(QMessageBox.Critical)
        dialog.exec_()
    def show_dialog2(self):
        dialog = QMessageBox(MainWindow)
        dialog.setText('Model khong dung')
        dialog.setIcon(QMessageBox.Critical)
        dialog.exec_()
    def show_dialog3(self):
        dialog = QMessageBox(MainWindow)
        dialog.setText(' Khong co du lieu')
        dialog.setIcon(QMessageBox.Critical)
        dialog.exec_()
    def show_dialog4(self):
        dialog = QMessageBox(MainWindow)
        dialog.setText(' Luu thanh cong')
        dialog.setIcon(QMessageBox.Critical)
        dialog.exec_()
    def show_dialog5(self):
        dialog = QMessageBox(MainWindow)
        dialog.setText('Lay mau thanh cong')
        dialog.setIcon(QMessageBox.Critical)
        dialog.exec_()
    def update_fun(self):
    
        connection = mysql.connector.connect(host='localhost',
                                            database = 'CAMERA_PAPER',
                                            user ='vqbg',
                                            password='vqbg123!')
        try:              
            val_update=(int(self.runmodel),(int(self.pictureid)),int(self.qtycam1),int(self.qtycam2), int(self.upperframe1), int(self.lowerframe1), int(self.leftframe1), int(self.rightframe1), int(self.contrast1),int(self.brightness1), int(self.adjustthreshold1), int(self.actualpixel1), int(self.setvalue1), int(self.adjustdifferent1), int(self.actualdifferent1), int(self.setdifferent1), int(self.contour1),int(self.setcontour1),int(self.actuallengh1), int(self.lenghmin1),int(self.lenghmax1), int(self.outerelipa1), int(self.outerelipb1), int(self.outerdir1),int(self.innerelipa1), int(self.innerelipb1),int(self.innerdir1), int(self.widthglue1), int(self.differentthreshold1), int(self.anabrightness1), int(self.anacontrast1), int(self.kerneldilate1), int(self.kernelerode1), int(self.upperframe2), int(self.lowerframe2), int(self.leftframe2), int(self.rightframe2), int(self.contrast2),int(self.brightness2), int(self.adjustthreshold2), int(self.actualpixel2), int(self.setvalue2), int(self.adjustdifferent2), int(self.actualdifferent2), int(self.setdifferent2), int(self.contour2),int(self.setcontour2),int(self.actuallengh2), int(self.lenghmin2),int(self.lenghmax2), int(self.outerelipa2), int(self.outerelipb2), int(self.outerdir2),int(self.innerelipa2), int(self.innerelipb2),int(self.innerdir2), int(self.widthglue2), int(self.differentthreshold2), int(self.anabrightness2), int(self.anacontrast2), int(self.kerneldilate2), int(self.kernelerode2),int(self.runmodel),int(self.pictureid))
                    
            mySql_Update_Table_Query=  "UPDATE SETTING_DATA SET runmodel=%s,pictureid=%s,qtycam1=%s,qtycam2=%s,upperframe1=%s,lowerframe1=%s,leftframe1=%s,rightframe1=%s,contrast1=%s,brightness1=%s,adjustthreshold1=%s,actualpixel1=%s,setvalue1=%s,adjustdifferent1=%s,actualdifferent1=%s,setdifferent1=%s,contour1=%s,setcontour1= %s,actuallengh1=%s,lenghmin1=%s,lenghmax1=%s,outerelipa1=%s,outerelipb1=%s,outerdir1=%s,innerelipa1=%s,innerelipb1=%s,innerdir1=%s,widthglue1=%s,differentthreshold1=%s,anabrightness1=%s,anacontrast1= %s,kerneldilate1=%s,kernelerode1=%s,upperframe2=%s,lowerframe2=%s,leftframe2=%s,rightframe2=%s,contrast2=%s,brightness2=%s,adjustthreshold2=%s,actualpixel2=%s,setvalue2=%s,adjustdifferent2= %s,actualdifferent2=%s,setdifferent2=%s,contour2= %s,setcontour2=%s,actuallengh2=%s,lenghmin2=%s,lenghmax2=%s,outerelipa2=%s,outerelipb2=%s,outerdir2=%s,innerelipa2=%s,innerelipb2=%s,innerdir2= %s,widthglue2=%s,differentthreshold2=%s,anabrightness2=%s,anacontrast2=%s,kerneldilate2=%s,kernelerode2=%s WHERE runmodel=%s and pictureid= %s"
            cursor = connection.cursor()
            result_update = cursor.execute(mySql_Update_Table_Query,val_update)
            mySql_Select_Table_Query=  "DELETE FROM SAVE_DATA"
            cursor1 = connection.cursor()
            result_model = cursor1.execute(mySql_Select_Table_Query)
                    
            mySql_Copy_Table_Query="""INSERT INTO SAVE_DATA SELECT*FROM SETTING_DATA WHERE runmodel= %s"""
            val_insert=(int(self.runmodel),)
            cursor2 = connection.cursor()
            result_insert = cursor2.execute(mySql_Copy_Table_Query,val_insert)

            print("update OK")

            

            connection.commit()
            self.readdata_fun()
            print(globalvar.read_database)
        except mysql.connector.Error as error:
            print("Faile to update table in MySQL : {}".format(error))

    def readdata_fun(self):
    
        connection = mysql.connector.connect(host='localhost',
                                            database = 'CAMERA_PAPER',
                                            user ='vqbg',
                                            password='vqbg123!')
        try:
            mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA where runmodel = %s"
            val_select=(int(self.runmodel),)
            cursor = connection.cursor()
            result_model = cursor.execute(mySql_Select_Table_Query,val_select)
            rows=cursor.fetchall()
            row_count=cursor.fetchmany()
            globalvar.read_database = np.array(rows)
            #print(self.read_database)
            cursor.close()
            connection.close()              
        except mysql.connector.Error as error:
            print("Faile to update table in MySQL : {}".format(error))
    def createdatabtn_fun(self):
        #print("model:",self.model)
        try:
            connection = mysql.connector.connect(host='localhost',
                                                database = 'CAMERA_PAPER',
                                                user ='vqbg',
                                                password='vqbg123!')
            mySql_Creat_Table_Query = """ CREATE TABLE SETTING_DATA (
                                        runmodel int(11) NULL,
                                        pictureid int(11) NULL,
                                        qtycam1 int(11) NULL,
                                        qtycam2 int(11) NULL,
                                        upperframe1 int(11) NULL,
                                        lowerframe1 int(11) NULL,
                                        leftframe1 int(11) NULL,
                                        rightframe1 int(11) NULL,
                                        contrast1 int(11) NULL,
                                        brightness1 int(11) NULL,
                                        adjustthreshold1 int(11) NULL,
                                        actualpixel1 int(11) NULL,
                                        setvalue1 int(11) NULL,
                                        adjustdifferent1 int(11) NULL,
                                        actualdifferent1 int(11) NULL,
                                        setdifferent1 int(11) NULL,
                                        contour1 int(11) NULL,
                                        setcontour1 int(11) NULL,
                                        actuallengh1 int(11) NULL,
                                        lenghmin1 int(11) NULL,
                                        lenghmax1 int(11) NULL,
                                        outerelipa1 int(11) NULL,
                                        outerelipb1 int(11) NULL,
                                        outerdir1 int(11) NULL,
                                        innerelipa1 int(11) NULL,
                                        innerelipb1 int(11) NULL,
                                        innerdir1 int(11) NULL,
                                        widthglue1 int(11) NULL,
                                        differentthreshold1 int(11) NULL,
                                        anabrightness1 int(11) NULL,
                                        anacontrast1 int(11) NULL,
                                        kerneldilate1 int(11) NULL,
                                        kernelerode1 int(11) NULL,
                                        upperframe2 int(11) NULL,
                                        lowerframe2 int(11) NULL,
                                        leftframe2 int(11) NULL,
                                        rightframe2 int(11) NULL,
                                        contrast2 int(11) NULL,
                                        brightness2 int(11) NULL,
                                        adjustthreshold2 int(11) NULL,
                                        actualpixel2 int(11) NULL,
                                        setvalue2 int(11) NULL,
                                        adjustdifferent2 int(11) NULL,
                                        actualdifferent2 int(11) NULL,
                                        setdifferent2 int(11) NULL,
                                        contour2 int(11) NULL,
                                        setcontour2 int(11) NULL,
                                        actuallengh2 int(11) NULL,
                                        lenghmin2 int(11) NULL,
                                        lenghmax2 int(11) NULL,
                                        outerelipa2 int(11) NULL,
                                        outerelipb2 int(11) NULL,
                                        outerdir2 int(11) NULL,
                                        innerelipa2 int(11) NULL,
                                        innerelipb2 int(11) NULL,
                                        innerdir2 int(11) NULL,
                                        widthglue2 int(11) NULL,
                                        differentthreshold2 int(11) NULL,
                                        anabrightness2 int(11) NULL,
                                        anacontrast2 int(11) NULL,
                                        kerneldilate2 int(11) NULL,
                                        kernelerode2 int(11) NULL                            
                                        )"""
            cursor = connection.cursor()
            result = cursor.execute(mySql_Creat_Table_Query)

            print("Setting table create sucessfully")

        except mysql.connector.Error as error:
            print("Faile to creat table in MySQL : {}".format(error))
            
        try:
            connection = mysql.connector.connect(host='localhost',
                                                database = 'CAMERA_PAPER',
                                                user ='vqbg',
                                                password='vqbg123!')
            mySql_Creat_Table_Query = """ CREATE TABLE SAVE_DATA (
                                        runmodel int(11) NULL,
                                        pictureid int(11) NULL,
                                        qtycam1 int(11) NULL,
                                        qtycam2 int(11) NULL,
                                        upperframe1 int(11) NULL,
                                        lowerframe1 int(11) NULL,
                                        leftframe1 int(11) NULL,
                                        rightframe1 int(11) NULL,
                                        contrast1 int(11) NULL,
                                        brightness1 int(11) NULL,
                                        adjustthreshold1 int(11) NULL,
                                        actualpixel1 int(11) NULL,
                                        setvalue1 int(11) NULL,
                                        adjustdifferent1 int(11) NULL,
                                        actualdifferent1 int(11) NULL,
                                        setdifferent1 int(11) NULL,
                                        contour1 int(11) NULL,
                                        setcontour1 int(11) NULL,
                                        actuallengh1 int(11) NULL,
                                        lenghmin1 int(11) NULL,
                                        lenghmax1 int(11) NULL,
                                        outerelipa1 int(11) NULL,
                                        outerelipb1 int(11) NULL,
                                        outerdir1 int(11) NULL,
                                        innerelipa1 int(11) NULL,
                                        innerelipb1 int(11) NULL,
                                        innerdir1 int(11) NULL,
                                        widthglue1 int(11) NULL,
                                        differentthreshold1 int(11) NULL,
                                        anabrightness1 int(11) NULL,
                                        anacontrast1 int(11) NULL,
                                        kerneldilate1 int(11) NULL,
                                        kernelerode1 int(11) NULL,
                                        upperframe2 int(11) NULL,
                                        lowerframe2 int(11) NULL,
                                        leftframe2 int(11) NULL,
                                        rightframe2 int(11) NULL,
                                        contrast2 int(11) NULL,
                                        brightness2 int(11) NULL,
                                        adjustthreshold2 int(11) NULL,
                                        actualpixel2 int(11) NULL,
                                        setvalue2 int(11) NULL,
                                        adjustdifferent2 int(11) NULL,
                                        actualdifferent2 int(11) NULL,
                                        setdifferent2 int(11) NULL,
                                        contour2 int(11) NULL,
                                        setcontour2 int(11) NULL,
                                        actuallengh2 int(11) NULL,
                                        lenghmin2 int(11) NULL,
                                        lenghmax2 int(11) NULL,
                                        outerelipa2 int(11) NULL,
                                        outerelipb2 int(11) NULL,
                                        outerdir2 int(11) NULL,
                                        innerelipa2 int(11) NULL,
                                        innerelipb2 int(11) NULL,
                                        innerdir2 int(11) NULL,
                                        widthglue2 int(11) NULL,
                                        differentthreshold2 int(11) NULL,
                                        anabrightness2 int(11) NULL,
                                        anacontrast2 int(11) NULL,
                                        kerneldilate2 int(11) NULL,
                                        kernelerode2 int(11) NULL                            
                                        )"""
            cursor = connection.cursor()
            result = cursor.execute(mySql_Creat_Table_Query)

            print("Setting table create sucessfully")

        except mysql.connector.Error as error:
            print("Faile to creat table in MySQL : {}".format(error))
    def deletedatabtn_fun(self):
        if (self.runmodel>600000):
            
            try:
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                #connection.commit()
                mySql_Select_Table_Query=  "DELETE FROM SETTING_DATA where runmodel = %s"
                val_select=(int(self.runmodel),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query,val_select)
  
                connection.commit()
                print("DELETE OK")
            except mysql.connector.Error as error:
                print("Faile to creat table in MySQL : {}".format(error))                
        else:
            print("No Model set")
        return
            
    def loaddata(self):
        
        if (self.runmodel>600000):
            
            try:
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                #connection.commit()
                mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA where runmodel = %s"
                val_select=(int(self.runmodel),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query,val_select)
                model_data=cursor.fetchall()
                row_count=cursor.fetchmany()
                if cursor.rowcount > 0:
                    mySql_Select_Table_Query=  "DELETE FROM SAVE_DATA"
                    cursor1 = connection.cursor()
                    result_model = cursor1.execute(mySql_Select_Table_Query)
                    
                    mySql_Copy_Table_Query="""INSERT INTO SAVE_DATA SELECT*FROM SETTING_DATA WHERE runmodel= %s"""
                    val_insert=(int(self.runmodel),)
                    cursor2 = connection.cursor()
                    result_insert = cursor2.execute(mySql_Copy_Table_Query,val_insert)
                                
                    for row in model_data:
                        #self.pictureid_check =row[1]
                        self.pictureid =row[1]
                        self.qtycam1 =row[2]                  
                        self.qtycam2 =row[3]                   
                        self.upperframe1=row[4]
                        self.lowerframe1=row[5]                    
                        self.leftframe1 =row[6]                   
                        self.rightframe1 =row[7]                   
                        self.contrast1 =row[8]                   
                        self.brightness1 =row[9]                  
                        self.adjustthreshold1=row[10]                   
                        self.actualpixel1=row[11]                    
                        self.setvalue1=row[12]
                        self.adjustdifferent1=row[13]
                        self.actualdifferent1=row[14]
                        self.setdifferent1=row[15]
                        self.contour1=row[16]
                        self.setcontour1=row[17]
                        self.actuallengh1=row[18]
                        self.lenghmin1=row[19]
                        self.lenghmax1=row[20]
                        self.outerelipa1=row[21]
                        self.outerelipb1=row[22]
                        self.outerdir1=row[23]
                        self.innerelipa1=row[24]
                        self.innerelipb1=row[25]
                        self.innerdir1=row[26]
                        self.widthglue1=row[27]
                        self.differentthreshold1=row[28]
                        self.anabrightness1=row[29]
                        self.anacontrast1=row[30]
                        self.kerneldilate1=row[31]
                        self.kernelerode1=row[32]
                        #video2
                        self.upperframe2=row[33]
                        self.lowerframe2=row[34]                    
                        self.leftframe2 =row[35]                   
                        self.rightframe2 =row[36]                   
                        self.contrast2 =row[37]                   
                        self.brightness2 =row[38]                  
                        self.adjustthreshold2=row[39]                   
                        self.actualpixel2=row[40]                    
                        self.setvalue2=row[41]
                        self.adjustdifferent2=row[42]
                        self.actualdifferent2=row[43]
                        self.setdifferent2=row[44]
                        self.contour2=row[45]
                        self.setcontour2=row[46]
                        self.actuallengh2=row[47]
                        self.lenghmin2=row[48]
                        self.lenghmax2=row[49]
                        self.outerelipa2=row[50]
                        self.outerelipb2=row[51]
                        self.outerdir2=row[52]
                        self.innerelipa2=row[53]
                        self.innerelipb2=row[54]
                        self.innerdir2=row[55]
                        self.widthglue2=row[56]
                        self.differentthreshold2=row[57]
                        self.anabrightness2=row[58]
                        self.anacontrast2=row[59]
                        self.kerneldilate2=row[60]
                        self.kernelerode2=row[61]
                        self.pictureidbox.setValue(self.pictureid)
                
                        self.qtycam1box.setValue(self.qtycam1)
                        self.qtycam2box.setValue(self.qtycam2)
                        self.pictureidbox.setValue(self.pictureid)
                        self.upperframe1box.setValue(self.upperframe1)
                        self.lowerframe1box.setValue(self.lowerframe1)
                        self.leftframe1box.setValue(self.leftframe1)
                        self.rightframe1box.setValue(self.rightframe1)
                        self.contrast1box.setValue(self.contrast1)
                        self.brightness1box.setValue(self.brightness1)
                        self.adjustthreshold1box.setValue(self.adjustthreshold1)
                        self.actualpixel1box.setValue(self.actualpixel1)
                        self.setvalue1box.setValue(self.setvalue1)
                        self.adjustdifferent1box.setValue(self.adjustdifferent1)
                        self.actualdifferent1box.setValue(self.actualdifferent1)
                        self.setdifferent1box.setValue(self.setdifferent1)
                        self.contour1box.setValue(self.contour1)
                        self.setcontour1box.setValue(self.setcontour1)
                        self.actuallengh1box.setValue(self.actuallengh1)
                        self.lenghmin1box.setValue(self.lenghmin1)
                        self.lenghmax1box.setValue(self.lenghmax1)
                        self.outerelipa1box.setValue(self.outerelipa1)
                        self.outerelipb1box.setValue(self.outerelipb1)
                        self.outerdir1box.setValue(self.outerdir1)
                        self.innerelipa1box.setValue(self.innerelipa1)
                        self.innerelipb1box.setValue(self.innerelipb1)
                        self.innerdir1box.setValue(self.innerdir1)
                        self.widthglue1box.setValue(self.widthglue1)
                        self.differentthreshold1box.setValue(self.differentthreshold1)
                        self.anabrightness1box.setValue(self.anabrightness1)
                        self.anacontrast1box.setValue(self.anacontrast1)
                        self.kerneldilate1box.setValue(self.kerneldilate1)
                        self.kernelerode1box.setValue(self.kernelerode1)

                        self.upperframe2box.setValue(self.upperframe2)
                        self.lowerframe2box.setValue(self.lowerframe2)
                        self.leftframe2box.setValue(self.leftframe2)
                        self.rightframe2box.setValue(self.rightframe2)
                        self.contrast2box.setValue(self.contrast2)
                        self.brightness2box.setValue(self.brightness2)
                        self.adjustthreshold2box.setValue(self.adjustthreshold2)
                        self.actualpixel2box.setValue(self.actualpixel2)
                        self.setvalue2box.setValue(self.setvalue2)
                        self.adjustdifferent2box.setValue(self.adjustdifferent2)
                        self.actualdifferent2box.setValue(self.actualdifferent2)
                        self.setdifferent2box.setValue(self.setdifferent2)
                        self.contour2box.setValue(self.contour2)
                        self.setcontour2box.setValue(self.setcontour2)
                        self.actuallengh2box.setValue(self.actuallengh2)
                        self.lenghmin2box.setValue(self.lenghmin2)
                        self.lenghmax2box.setValue(self.lenghmax2)
                        self.outerelipa2box.setValue(self.outerelipa2)
                        self.outerelipb2box.setValue(self.outerelipb2)
                        self.outerdir2box.setValue(self.outerdir2)
                        self.innerelipa2box.setValue(self.innerelipa2)
                        self.innerelipb2box.setValue(self.innerelipb2)
                        self.innerdir2box.setValue(self.innerdir2)
                        self.widthglue2box.setValue(self.widthglue2)
                        self.differentthreshold2box.setValue(self.differentthreshold2)
                        self.anabrightness2box.setValue(self.anabrightness2)
                        self.anacontrast2box.setValue(self.anacontrast2)
                        self.kerneldilate2box.setValue(self.kerneldilate2)
                        self.kernelerode2box.setValue(self.kernelerode2)
                        #print("LOAD DATA OK")

                else:
                    self.show_dialog3()  
                connection.commit()
            except mysql.connector.Error as error:
                print("Faile to creat table in MySQL : {}".format(error))                
        else:
            print("Faile data")
        return

    def loaddata_start(self):
        #self.startstart = 0
        if (self.start == 0):       
            try:
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                #connection.commit()
                mySql_Select_Table_Query=  "SELECT * FROM SAVE_DATA"
                #val_select=(int(self.runmodel),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query)
                model_data=cursor.fetchall()
                row_count=cursor.fetchmany()
                if cursor.rowcount > 0:                                
                    for row in model_data:
                        #self.pictureid_check =row[1]
                        self.runmodel =row[0]
                        self.pictureid =row[1]
                        self.qtycam1 =row[2]                  
                        self.qtycam2 =row[3]                   
                        self.upperframe1=row[4]
                        self.lowerframe1=row[5]                    
                        self.leftframe1 =row[6]                   
                        self.rightframe1 =row[7]                   
                        self.contrast1 =row[8]                   
                        self.brightness1 =row[9]                  
                        self.adjustthreshold1=row[10]                   
                        self.actualpixel1=row[11]                    
                        self.setvalue1=row[12]
                        self.adjustdifferent1=row[13]
                        self.actualdifferent1=row[14]
                        self.setdifferent1=row[15]
                        self.contour1=row[16]
                        self.setcontour1=row[17]
                        self.actuallengh1=row[18]
                        self.lenghmin1=row[19]
                        self.lenghmax1=row[20]
                        self.outerelipa1=row[21]
                        self.outerelipb1=row[22]
                        self.outerdir1=row[23]
                        self.innerelipa1=row[24]
                        self.innerelipb1=row[25]
                        self.innerdir1=row[26]
                        self.widthglue1=row[27]
                        self.differentthreshold1=row[28]
                        self.anabrightness1=row[29]
                        self.anacontrast1=row[30]
                        self.kerneldilate1=row[31]
                        self.kernelerode1=row[32]
                        #video2
                        self.upperframe2=row[33]
                        self.lowerframe2=row[34]                    
                        self.leftframe2 =row[35]                   
                        self.rightframe2 =row[36]                   
                        self.contrast2 =row[37]                   
                        self.brightness2 =row[38]                  
                        self.adjustthreshold2=row[39]                   
                        self.actualpixel2=row[40]                    
                        self.setvalue2=row[41]
                        self.adjustdifferent2=row[42]
                        self.actualdifferent2=row[43]
                        self.setdifferent2=row[44]
                        self.contour2=row[45]
                        self.setcontour2=row[46]
                        self.actuallengh2=row[47]
                        self.lenghmin2=row[48]
                        self.lenghmax2=row[49]
                        self.outerelipa2=row[50]
                        self.outerelipb2=row[51]
                        self.outerdir2=row[52]
                        self.innerelipa2=row[53]
                        self.innerelipb2=row[54]
                        self.innerdir2=row[55]
                        self.widthglue2=row[56]
                        self.differentthreshold2=row[57]
                        self.anabrightness2=row[58]
                        self.anacontrast2=row[59]
                        self.kerneldilate2=row[60]
                        self.kernelerode2=row[61]
                        self.runmodelbox.setValue(self.runmodel)
                        self.pictureidbox.setValue(self.pictureid)
                
                        self.qtycam1box.setValue(self.qtycam1)
                        self.qtycam2box.setValue(self.qtycam2)
                        self.pictureidbox.setValue(self.pictureid)
                        self.upperframe1box.setValue(self.upperframe1)
                        self.lowerframe1box.setValue(self.lowerframe1)
                        self.leftframe1box.setValue(self.leftframe1)
                        self.rightframe1box.setValue(self.rightframe1)
                        self.contrast1box.setValue(self.contrast1)
                        self.brightness1box.setValue(self.brightness1)
                        self.adjustthreshold1box.setValue(self.adjustthreshold1)
                        self.actualpixel1box.setValue(self.actualpixel1)
                        self.setvalue1box.setValue(self.setvalue1)
                        self.adjustdifferent1box.setValue(self.adjustdifferent1)
                        self.actualdifferent1box.setValue(self.actualdifferent1)
                        self.setdifferent1box.setValue(self.setdifferent1)
                        self.contour1box.setValue(self.contour1)
                        self.setcontour1box.setValue(self.setcontour1)
                        self.actuallengh1box.setValue(self.actuallengh1)
                        self.lenghmin1box.setValue(self.lenghmin1)
                        self.lenghmax1box.setValue(self.lenghmax1)
                        self.outerelipa1box.setValue(self.outerelipa1)
                        self.outerelipb1box.setValue(self.outerelipb1)
                        self.outerdir1box.setValue(self.outerdir1)
                        self.innerelipa1box.setValue(self.innerelipa1)
                        self.innerelipb1box.setValue(self.innerelipb1)
                        self.innerdir1box.setValue(self.innerdir1)
                        self.widthglue1box.setValue(self.widthglue1)
                        self.differentthreshold1box.setValue(self.differentthreshold1)
                        self.anabrightness1box.setValue(self.anabrightness1)
                        self.anacontrast1box.setValue(self.anacontrast1)
                        self.kerneldilate1box.setValue(self.kerneldilate1)
                        self.kernelerode1box.setValue(self.kernelerode1)

                        self.upperframe2box.setValue(self.upperframe2)
                        self.lowerframe2box.setValue(self.lowerframe2)
                        self.leftframe2box.setValue(self.leftframe2)
                        self.rightframe2box.setValue(self.rightframe2)
                        self.contrast2box.setValue(self.contrast2)
                        self.brightness2box.setValue(self.brightness2)
                        self.adjustthreshold2box.setValue(self.adjustthreshold2)
                        self.actualpixel2box.setValue(self.actualpixel2)
                        self.setvalue2box.setValue(self.setvalue2)
                        self.adjustdifferent2box.setValue(self.adjustdifferent2)
                        self.actualdifferent2box.setValue(self.actualdifferent2)
                        self.setdifferent2box.setValue(self.setdifferent2)
                        self.contour2box.setValue(self.contour2)
                        self.setcontour2box.setValue(self.setcontour2)
                        self.actuallengh2box.setValue(self.actuallengh2)
                        self.lenghmin2box.setValue(self.lenghmin2)
                        self.lenghmax2box.setValue(self.lenghmax2)
                        self.outerelipa2box.setValue(self.outerelipa2)
                        self.outerelipb2box.setValue(self.outerelipb2)
                        self.outerdir2box.setValue(self.outerdir2)
                        self.innerelipa2box.setValue(self.innerelipa2)
                        self.innerelipb2box.setValue(self.innerelipb2)
                        self.innerdir2box.setValue(self.innerdir2)
                        self.widthglue2box.setValue(self.widthglue2)
                        self.differentthreshold2box.setValue(self.differentthreshold2)
                        self.anabrightness2box.setValue(self.anabrightness2)
                        self.anacontrast2box.setValue(self.anacontrast2)
                        self.kerneldilate2box.setValue(self.kerneldilate2)
                        self.kernelerode2box.setValue(self.kernelerode2)
                        #print("LOAD DATA OK")
                        self.readdata_fun()
                        self.start = 1
                        print(self.read_database)

                else:
                    self.show_dialog3()
                  
                connection.commit()
            except mysql.connector.Error as error:
                print("Faile to creat table in MySQL : {}".format(error))                
        else:
            print("Faile data")
        return
        
    def runmodelbox_fun(self,value):
        self.runmodel = value
        globalvar.runmodel = value

        self.start=0
        self.loaddata()
        
    def copydatabtn_fun(self):
        if (self.runmodel >600000)&(self.pictureid >0)&(self.copymodel >600000):
            try:
                
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA where runmodel = %s"
                val_select=(int(self.copymodel),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query,val_select)
                row_count=cursor.fetchmany()
                if cursor.rowcount <= 0:
                    
                    mySql_Select_Table_Query=  "DELETE FROM SAVE_DATA"
                    cursor1 = connection.cursor()
                    result_model = cursor1.execute(mySql_Select_Table_Query)
                
                    mySql_Copy_Table_Query="""INSERT INTO SAVE_DATA SELECT*FROM SETTING_DATA WHERE runmodel= %s"""
                    val_insert=(int(self.runmodel),)
                    cursor2 = connection.cursor()
                    result_insert = cursor2.execute(mySql_Copy_Table_Query,val_insert)
                
                    mySql_update_Table_Query="""UPDATE SAVE_DATA SET runmodel = %s WHERE runmodel = %s"""
                    val_update=(int(self.copymodel),int(self.runmodel),)
                    cursor3 = connection.cursor()
                    result_insert = cursor3.execute(mySql_update_Table_Query,val_update)
                
                
                    mySql_Copy_Table_Query="""INSERT INTO SETTING_DATA SELECT*FROM SAVE_DATA WHERE runmodel= %s"""
                    val_insert=(int(self.copymodel),)
                    cursor4 = connection.cursor()
                    result_insert = cursor4.execute(mySql_Copy_Table_Query,val_insert)
                
                    mySql_Select_Table_Query=  "DELETE FROM SAVE_DATA"
                    cursor5 = connection.cursor()
                    result_model = cursor5.execute(mySql_Select_Table_Query)
                    
                    mySql_Copy_Table_Query="""INSERT INTO SAVE_DATA SELECT*FROM SETTING_DATA WHERE runmodel= %s"""
                    val_insert=(int(self.runmodel),)
                    cursor2 = connection.cursor()
                    result_insert = cursor2.execute(mySql_Copy_Table_Query,val_insert)
                    

                    self.show_dialog4()
                else:
                    self.show_dialog1()
                
                connection.commit()
    
            except mysql.connector.Error as error:
                print("Faile to create table in MySQL : {}".format(error))           
        else:
            self.show_dialog2()
    
    def copymodelbox_fun(self,value):
        self.copymodel = value
    def qtycam1box_fun(self,value):
        self.qtycam1 = value
        globalvar.qtycam1 = value
        if self.start == 1:
            self.update_fun()
    def qtycam2box_fun(self,value):
        self.qtycam2 = value
        globalvar.qtycam2 = value
        if self.start ==1:
            self.update_fun()
    def pictureidbox_fun(self,value):
        self.pictureid = value
        globalvar.pictureid = value
      
        if (self.runmodel>600000)&(self.pictureid>0):
            try:
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                #connection.commit()
                mySql_Select_Table_Query=  "SELECT * FROM SETTING_DATA where runmodel = %s and pictureid= %s"
                val_select=(int(self.runmodel),int(self.pictureid),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query,val_select)
                model_data=cursor.fetchall()
                row_count=cursor.fetchmany()
                if cursor.rowcount > 0:            
                    for row in model_data:
                        self.pictureid_check =row[1]
                        self.qtycam1 =row[2]                  
                        self.qtycam2 =row[3]                   
                        self.upperframe1=row[4]
                        self.lowerframe1=row[5]                    
                        self.leftframe1 =row[6]                   
                        self.rightframe1 =row[7]                   
                        self.contrast1 =row[8]                   
                        self.brightness1 =row[9]                  
                        self.adjustthreshold1=row[10]                   
                        self.actualpixel1=row[11]                    
                        self.setvalue1=row[12]
                        self.adjustdifferent1=row[13]
                        self.actualdifferent1=row[14]
                        self.setdifferent1=row[15]
                        self.contour1=row[16]
                        self.setcontour1=row[17]
                        self.actuallengh1=row[18]
                        self.lenghmin1=row[19]
                        self.lenghmax1=row[20]
                        self.outerelipa1=row[21]
                        self.outerelipb1=row[22]
                        self.outerdir1=row[23]
                        self.innerelipa1=row[24]
                        self.innerelipb1=row[25]
                        self.innerdir1=row[26]
                        self.widthglue1=row[27]
                        self.differentthreshold1=row[28]
                        self.anabrightness1=row[29]
                        self.anacontrast1=row[30]
                        self.kerneldilate1=row[31]
                        self.kernelerode1=row[32]
                        #video2
                        self.upperframe2=row[33]
                        self.lowerframe2=row[34]                    
                        self.leftframe2 =row[35]                   
                        self.rightframe2 =row[36]                   
                        self.contrast2 =row[37]                   
                        self.brightness2 =row[38]                  
                        self.adjustthreshold2=row[39]                   
                        self.actualpixel2=row[40]                    
                        self.setvalue2=row[41]
                        self.adjustdifferent2=row[42]
                        self.actualdifferent2=row[43]
                        self.setdifferent2=row[44]
                        self.contour2=row[45]
                        self.setcontour2=row[46]
                        self.actuallengh2=row[47]
                        self.lenghmin2=row[48]
                        self.lenghmax2=row[49]
                        self.outerelipa2=row[50]
                        self.outerelipb2=row[51]
                        self.outerdir2=row[52]
                        self.innerelipa2=row[53]
                        self.innerelipb2=row[54]
                        self.innerdir2=row[55]
                        self.widthglue2=row[56]
                        self.differentthreshold2=row[57]
                        self.anabrightness2=row[58]
                        self.anacontrast2=row[59]
                        self.kerneldilate2=row[60]
                        self.kernelerode2=row[61]
                
                        self.qtycam1box.setValue(self.qtycam1)
                        self.qtycam2box.setValue(self.qtycam2)
                        self.upperframe1box.setValue(self.upperframe1)
                        self.lowerframe1box.setValue(self.lowerframe1)
                        self.leftframe1box.setValue(self.leftframe1)
                        self.rightframe1box.setValue(self.rightframe1)
                        self.contrast1box.setValue(self.contrast1)
                        self.brightness1box.setValue(self.brightness1)
                        self.adjustthreshold1box.setValue(self.adjustthreshold1)
                        self.actualpixel1box.setValue(self.actualpixel1)
                        self.setvalue1box.setValue(self.setvalue1)
                        self.adjustdifferent1box.setValue(self.adjustdifferent1)
                        self.actualdifferent1box.setValue(self.actualdifferent1)
                        self.setdifferent1box.setValue(self.setdifferent1)
                        self.contour1box.setValue(self.contour1)
                        self.setcontour1box.setValue(self.setcontour1)
                        self.actuallengh1box.setValue(self.actuallengh1)
                        self.lenghmin1box.setValue(self.lenghmin1)
                        self.lenghmax1box.setValue(self.lenghmax1)
                        self.outerelipa1box.setValue(self.outerelipa1)
                        self.outerelipb1box.setValue(self.outerelipb1)
                        self.outerdir1box.setValue(self.outerdir1)
                        self.innerelipa1box.setValue(self.innerelipa1)
                        self.innerelipb1box.setValue(self.innerelipb1)
                        self.innerdir1box.setValue(self.innerdir1)
                        self.widthglue1box.setValue(self.widthglue1)
                        self.differentthreshold1box.setValue(self.differentthreshold1)
                        self.anabrightness1box.setValue(self.anabrightness1)
                        self.anacontrast1box.setValue(self.anacontrast1)
                        self.kerneldilate1box.setValue(self.kerneldilate1)
                        self.kernelerode1box.setValue(self.kernelerode1)

                        self.upperframe2box.setValue(self.upperframe2)
                        self.lowerframe2box.setValue(self.lowerframe2)
                        self.leftframe2box.setValue(self.leftframe2)
                        self.rightframe2box.setValue(self.rightframe2)
                        self.contrast2box.setValue(self.contrast2)
                        self.brightness2box.setValue(self.brightness2)
                        self.adjustthreshold2box.setValue(self.adjustthreshold2)
                        self.actualpixel2box.setValue(self.actualpixel2)
                        self.setvalue2box.setValue(self.setvalue2)
                        self.adjustdifferent2box.setValue(self.adjustdifferent2)
                        self.actualdifferent2box.setValue(self.actualdifferent2)
                        self.setdifferent2box.setValue(self.setdifferent2)
                        self.contour2box.setValue(self.contour2)
                        self.setcontour2box.setValue(self.setcontour2)
                        self.actuallengh2box.setValue(self.actuallengh2)
                        self.lenghmin2box.setValue(self.lenghmin2)
                        self.lenghmax2box.setValue(self.lenghmax2)
                        self.outerelipa2box.setValue(self.outerelipa2)
                        self.outerelipb2box.setValue(self.outerelipb2)
                        self.outerdir2box.setValue(self.outerdir2)
                        self.innerelipa2box.setValue(self.innerelipa2)
                        self.innerelipb2box.setValue(self.innerelipb2)
                        self.innerdir2box.setValue(self.innerdir2)
                        self.widthglue2box.setValue(self.widthglue2)
                        self.differentthreshold2box.setValue(self.differentthreshold2)
                        self.anabrightness2box.setValue(self.anabrightness2)
                        self.anacontrast2box.setValue(self.anacontrast2)
                        self.kerneldilate2box.setValue(self.kerneldilate2)
                        self.kernelerode2box.setValue(self.kernelerode2)

                else:
                    self.show_dialog3()  
                connection.commit()
            except mysql.connector.Error as error:
                print("Faile to creat table in MySQL : {}".format(error))                
        else:
            self.show_dialog2()
        return self.pictureid_check
    # ham xu ly video 1
    # xu ly thread camera 1
    def startcam1btn_fun(self):
        self.thread1 = camera1(globalvar.read_database,globalvar.pictureid,globalvar.runmodel,globalvar.qtycam1,globalvar.save_picture_1)
        self.thread1.camera1.connect(self.show_cam1)
        self.thread1.anacamera1.connect(self.show_anacamera1)
        self.thread1.send_actualpixel1.connect(self.show_actualpixel1)
        self.thread1.send_actualdifferent1.connect(self.show_actualdifferent1) 
        #self.thread1.send_contour1.connect(self.show_contour1)
        #self.thread1.send_actuallengh1.connect(self.show_actuallengh1)
        self.thread1.start()
    def show_cam1(self,camera1):
        pixmap = QtGui.QPixmap.fromImage(camera1)
        self.camera01frame.setPixmap(pixmap) 
    def show_anacamera1(self,anacamera1):
        pixmap = QtGui.QPixmap.fromImage(anacamera1)
        self.anacam1frame.setPixmap(pixmap)
    def show_actualpixel1(self,actualpixel1):   
        self.actualpixel1box.setValue(actualpixel1)
    def show_actualdifferent1(self,actualdifferent1):
        self.actualdifferent1box.setValue(actualdifferent1)
    #def show_contour1(self,contour1):
        #self.contour1box.setValue(contour1)
    #def show_actuallengh1(self,actuallengh1):
        #self.actuallengh1box.setValue(actuallengh1)
   #Ham input data video 1
    def upperframe1box_fun(self,value):
        self.upperframe1 = value
        #start = self.loaddata_start()
        #print(self.start)
        if self.start ==1:
            self.update_fun()
            
            
    def lowerframe1box_fun(self,value):
        self.lowerframe1 = value
        if self.start ==1:
            self.update_fun()        
    def leftframe1box_fun(self,value):
        self.leftframe1 = value
        if self.start ==1:
            self.update_fun()
    def rightframe1box_fun(self,value):
        self.rightframe1 = value
        if self.start ==1:
            self.update_fun()
    def contrast1box_fun(self,value):
        self.contrast1 = value
        if self.start ==1:
            self.update_fun()
    def brightness1box_fun(self,value):
        self.brightness1 = value
        if self.start ==1:
            self.update_fun()
    def savedata1btn_fun(self):
        self.start=1
        if (self.runmodel >600000)&(self.pictureid >0):
            try:
                connection = mysql.connector.connect(host='localhost',
                                                    database = 'CAMERA_PAPER',
                                                    user ='vqbg',
                                                     password='vqbg123!')
                #connection.commit()
                mySql_Select_Table_Query=  "SELECT runmodel FROM SETTING_DATA where runmodel = %s and pictureid= %s"
                val_select=(int(self.runmodel),int(self.pictureid),)
                cursor = connection.cursor()
                result_model = cursor.execute(mySql_Select_Table_Query,val_select)
                row_count=cursor.fetchmany()
                if cursor.rowcount <=0:
                    mySql_Insert_Table_Query="""INSERT INTO SETTING_DATA (runmodel,pictureid,qtycam1,qtycam2,upperframe1,lowerframe1,leftframe1,rightframe1,contrast1,brightness1,adjustthreshold1,actualpixel1,setvalue1,adjustdifferent1,actualdifferent1,setdifferent1,contour1,setcontour1,actuallengh1,lenghmin1,lenghmax1,outerelipa1,outerelipb1,outerdir1,innerelipa1,innerelipb1,innerdir1,widthglue1,differentthreshold1,anabrightness1,anacontrast1,kerneldilate1,kernelerode1,upperframe2,lowerframe2,leftframe2,rightframe2,contrast2,brightness2,adjustthreshold2,actualpixel2,setvalue2,adjustdifferent2,actualdifferent2,setdifferent2,contour2,setcontour2,actuallengh2,lenghmin2,lenghmax2,outerelipa2,outerelipb2,outerdir2,innerelipa2,innerelipb2,innerdir2,widthglue2,differentthreshold2,anabrightness2,anacontrast2,kerneldilate2,kernelerode2) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                    #columns = [column[0] for column in cursor.fetchall()]
                    #mySql_Insert_Table_Query= f"INSERT INTO SETTING_DATA ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
                    
                    val_insert=(int(self.runmodel),(int(self.pictureid)),int(self.qtycam1),int(self.qtycam2), int(self.upperframe1), int(self.lowerframe1), int(self.leftframe1), int(self.rightframe1), int(self.contrast1),int(self.brightness1), int(self.adjustthreshold1), int(self.actualpixel1), int(self.setvalue1), int(self.adjustdifferent1), int(self.actualdifferent1), int(self.setdifferent1), int(self.contour1),int(self.setcontour1),int(self.actuallengh1), int(self.lenghmin1),int(self.lenghmax1), int(self.outerelipa1), int(self.outerelipb1), int(self.outerdir1),int(self.innerelipa1), int(self.innerelipb1),int(self.innerdir1), int(self.widthglue1), int(self.differentthreshold1), int(self.anabrightness1), int(self.anacontrast1), int(self.kerneldilate1), int(self.kernelerode1), int(self.upperframe2), int(self.lowerframe2), int(self.leftframe2), int(self.rightframe2), int(self.contrast2),int(self.brightness2), int(self.adjustthreshold2), int(self.actualpixel2), int(self.setvalue2), int(self.adjustdifferent2), int(self.actualdifferent2), int(self.setdifferent2), int(self.contour2),int(self.setcontour2),int(self.actuallengh2), int(self.lenghmin2),int(self.lenghmax2), int(self.outerelipa2), int(self.outerelipb2), int(self.outerdir2),int(self.innerelipa2), int(self.innerelipb2),int(self.innerdir2), int(self.widthglue2), int(self.differentthreshold2), int(self.anabrightness2), int(self.anacontrast2), int(self.kerneldilate2), int(self.kernelerode2))
                    #print("miss1")
                    #result_insert = cursor.execute(mySql_Insert_Table_Query,val_insert_1)
                    result_insert = cursor.execute(mySql_Insert_Table_Query,val_insert)
                    self.show_dialog4()
                    connection.commit()
                else:
                    val_update=(int(self.runmodel),(int(self.pictureid)),int(self.qtycam1),int(self.qtycam2), int(self.upperframe1), int(self.lowerframe1), int(self.leftframe1), int(self.rightframe1), int(self.contrast1),int(self.brightness1), int(self.adjustthreshold1), int(self.actualpixel1), int(self.setvalue1), int(self.adjustdifferent1), int(self.actualdifferent1), int(self.setdifferent1), int(self.contour1),int(self.setcontour1),int(self.actuallengh1), int(self.lenghmin1),int(self.lenghmax1), int(self.outerelipa1), int(self.outerelipb1), int(self.outerdir1),int(self.innerelipa1), int(self.innerelipb1),int(self.innerdir1), int(self.widthglue1), int(self.differentthreshold1), int(self.anabrightness1), int(self.anacontrast1), int(self.kerneldilate1), int(self.kernelerode1), int(self.upperframe2), int(self.lowerframe2), int(self.leftframe2), int(self.rightframe2), int(self.contrast2),int(self.brightness2), int(self.adjustthreshold2), int(self.actualpixel2), int(self.setvalue2), int(self.adjustdifferent2), int(self.actualdifferent2), int(self.setdifferent2), int(self.contour2),int(self.setcontour2),int(self.actuallengh2), int(self.lenghmin2),int(self.lenghmax2), int(self.outerelipa2), int(self.outerelipb2), int(self.outerdir2),int(self.innerelipa2), int(self.innerelipb2),int(self.innerdir2), int(self.widthglue2), int(self.differentthreshold2), int(self.anabrightness2), int(self.anacontrast2), int(self.kerneldilate2), int(self.kernelerode2),int(self.runmodel),int(self.pictureid))
                    #mySql_Update_Table_Query=  f"UPDATE SETTING_DATA SET {', '.join([f'{column} =%s' for column in val_update.keys()])}"
                    mySql_Update_Table_Query=  "UPDATE SETTING_DATA SET runmodel=%s,pictureid=%s,qtycam1=%s,qtycam2=%s,upperframe1=%s,lowerframe1=%s,leftframe1=%s,rightframe1=%s,contrast1=%s,brightness1=%s,adjustthreshold1=%s,actualpixel1=%s,setvalue1=%s,adjustdifferent1=%s,actualdifferent1=%s,setdifferent1=%s,contour1=%s,setcontour1= %s,actuallengh1=%s,lenghmin1=%s,lenghmax1=%s,outerelipa1=%s,outerelipb1=%s,outerdir1=%s,innerelipa1=%s,innerelipb1=%s,innerdir1=%s,widthglue1=%s,differentthreshold1=%s,anabrightness1=%s,anacontrast1= %s,kerneldilate1=%s,kernelerode1=%s,upperframe2=%s,lowerframe2=%s,leftframe2=%s,rightframe2=%s,contrast2=%s,brightness2=%s,adjustthreshold2=%s,actualpixel2=%s,setvalue2=%s,adjustdifferent2= %s,actualdifferent2=%s,setdifferent2=%s,contour2= %s,setcontour2=%s,actuallengh2=%s,lenghmin2=%s,lenghmax2=%s,outerelipa2=%s,outerelipb2=%s,outerdir2=%s,innerelipa2=%s,innerelipb2=%s,innerdir2= %s,widthglue2=%s,differentthreshold2=%s,anabrightness2=%s,anacontrast2=%s,kerneldilate2=%s,kernelerode2=%s WHERE runmodel=%s and pictureid= %s"
                    result_update = cursor.execute(mySql_Update_Table_Query,val_update)
                    connection.commit()
                    self.show_dialog4()
            except mysql.connector.Error as error:
                print("Faile to create table in MySQL : {}".format(error))           
        else:
            self.show_dialog2()
    def savesample1btn_fun(self):
        if globalvar.save_picture_1==0:
            globalvar.save_picture_1=1
            sleep(1)       
            globalvar.save_picture_1=0
            
    # Ham phan tich video 1
    def adjustthreshold1box_fun(self,value):
        self.adjustthreshold1 = value
        if self.start ==1:
            self.update_fun()
    def actualpixel1box_fun(self,value):
        self.actualpixel1 = value
        #if self.start ==1:
            #self.update_fun()
    def setvalue1box_fun(self,value):
        self.setvalue1 = value
        if self.start ==1:
            self.update_fun()
    def adjustdifferent1box_fun(self,value):
        self.adjustdifferent1 = value
        if self.start ==1:
            self.update_fun()
    def actualdifferent1box_fun(self,value):
        self.actualdifferent1 = value
        #if self.start ==1:
            #self.update_fun()
    def setdifferent1box_fun(self,value):
        self.setdifferent1 = value
        if self.start ==1:
            self.update_fun()
    def contour1box_fun(self,value):
        self.contour1 = value
        #if self.start ==1:
            #self.update_fun()
    def setcontour1box_fun(self,value):
        self.setcontour1 = value
        if self.start ==1:
            self.update_fun()
    def actuallengh1box_fun(self,value):
        self.actuallengh1 = value
        #if self.start ==1:
            #self.update_fun()
    def lenghmin1box_fun(self,value):
        self.lenghmin1 = value
        if self.start ==1:
            self.update_fun()
    def lenghmax1box_fun(self,value):
        self.lenghmax1 = value
        if self.start ==1:
            self.update_fun()
    def outerelipa1box_fun(self,value):
        self.outerelipa1 = value
        if self.start ==1:
            self.update_fun()
    def outerelipb1box_fun(self,value):
        self.outerelipb1 = value
        if self.start ==1:
            self.update_fun()
    def outerdir1box_fun(self,value):
        self.outerdir1= value
        if self.start ==1:
            self.update_fun()
    def innerelipa1box_fun(self,value):
        self.innerelipa1 = value
        if self.start ==1:
            self.update_fun()
    def innerelipb1box_fun(self,value):
        self.innerelipb1 = value
        if self.start ==1:
            self.update_fun()
    def innerdir1box_fun(self,value):
        self.innerdir1 = value
        if self.start ==1:
            self.update_fun()
    def widthglue1box_fun(self,value):
        self.widthglue1 = value
        if self.start ==1:
            self.update_fun()
    def differentthreshold1box_fun(self,value):
        self.differentthreshold1 = value
        if self.start ==1:
            self.update_fun()
    def anabrightness1box_fun(self,value):
        self.anabrightness1 = value
        if self.start ==1:
            self.update_fun()
    def anacontrast1box_fun(self,value):
        self.anacontrast1 = value
        if self.start ==1:
            self.update_fun()
    def kerneldilate1box_fun(self,value):
        self.kerneldilate1 = value
        if self.start ==1:
            self.update_fun()
    def kernelerode1box_fun(self,value):
        self.kernelerode1 = value
        if self.start ==1:
            self.update_fun()
    # Ham xu ly video 2
    def startcam2btn_fun(self):
        self.thread2 = camera2(globalvar.read_database,globalvar.pictureid,globalvar.runmodel,globalvar.qtycam1,globalvar.save_picture_2)
        self.thread2.camera2.connect(self.show_cam2)
        self.thread2.anacamera2.connect(self.show_anacamera2)
        self.thread2.send_actualpixel2.connect(self.show_actualpixel2)
        self.thread2.send_actualdifferent2.connect(self.show_actualdifferent2) 
        #self.thread2.send_contour2.connect(self.show_contour2)
        self.thread2.send_actuallengh2.connect(self.show_actuallengh2)
        self.thread2.start()
    def show_cam2(self,camera2):
        pixmap = QtGui.QPixmap.fromImage(camera2)
        self.camera02frame.setPixmap(pixmap) 
    def show_anacamera2(self,anacamera2):
        pixmap = QtGui.QPixmap.fromImage(anacamera2)
        self.anacam2frame.setPixmap(pixmap)
    def show_actualpixel2(self,actualpixel2):   
        self.actualpixel2box.setValue(actualpixel2)
    def show_actualdifferent2(self,actualdifferent2):
        self.actualdifferent2box.setValue(actualdifferent2)
    #def show_contour2(self,contour2):
        #self.contour2box.setValue(contour2)
    def show_actuallengh2(self,actuallengh2):
        self.actuallengh2box.setValue(actuallengh2)
    # ham xu ly da ta video 2
    def upperframe2box_fun(self,value):
        self.upperframe2 = value
        if self.start ==1:
            self.update_fun()
    def lowerframe2box_fun(self,value):
        self.lowerframe2 = value
        if self.start ==1:
            self.update_fun()
    def leftframe2box_fun(self,value):
        self.leftframe2 = value
        if self.start ==1:
            self.update_fun()
    def rightframe2box_fun(self,value):
        self.rightframe2 = value
        if self.start ==1:
            self.update_fun()
    def contrast2box_fun(self,value):
        self.contrast2 = value
        if self.start ==1:
            self.update_fun()
    def brightness2box_fun(self,value):
        self.brightness2 = value
        if self.start ==1:
            self.update_fun()
    #def savedata2btn_fun(self):


    
    def savesample2btn_fun(self):
        if globalvar.save_picture_2==0:
            globalvar.save_picture_2=1
            sleep(1)       
            globalvar.save_picture_2=0
    # Ham phan tich video 2
    def adjustthreshold2box_fun(self,value):
        self.adjustthreshold2 = value
        if self.start ==1:
            self.update_fun()
    def actualpixel2box_fun(self,value):
        self.actualpixel2 = value
        #if self.start ==1:
            #self.update_fun()
    def setvalue2box_fun(self,value):
        self.setvalue2 = value
        if self.start ==1:
            self.update_fun()
    def adjustdifferent2box_fun(self,value):
        self.adjustdifferent2 = value
        if self.start ==1:
            self.update_fun()
    def actualdifferent2box_fun(self,value):
        self.actualdifferent2 = value
        #if self.start ==1:
            #self.update_fun()
    def setdifferent2box_fun(self,value):
        self.setdifferent2 = value
        if self.start ==1:
            self.update_fun()
    def contour2box_fun(self,value):
        self.contour2 = value
        #if self.start ==1:
            #self.update_fun()
    def setcontour2box_fun(self,value):
        self.setcontour2 = value
        if self.start ==1:
            self.update_fun()
    def actuallengh2box_fun(self,value):
        self.actuallengh2 = value
        #if self.start ==1:
            #self.update_fun()
    def lenghmin2box_fun(self,value):
        self.lenghmin2 = value
        if self.start ==1:
            self.update_fun()
    def lenghmax2box_fun(self,value):
        self.lenghmax2 = value
        if self.start ==1:
            self.update_fun()
    def outerelipa2box_fun(self,value):
        self.outerelipa2 = value
        if self.start ==1:
            self.update_fun()
    def outerelipb2box_fun(self,value):
        self.outerelipb2 = value
        if self.start ==1:
            self.update_fun()
    def outerdir2box_fun(self,value):
        self.outerdir2 = value
        if self.start ==1:
            self.update_fun()
    def innerelipa2box_fun(self,value):
        self.innerelipa2 = value
        if self.start ==1:
            self.update_fun()
    def innerelipb2box_fun(self,value):
        self.innerelipb2 = value
        if self.start ==1:
            self.update_fun()
    def innerdir2box_fun(self,value):
        self.innerdir2 = value
        if self.start ==1:
            self.update_fun()
    def widthglue2box_fun(self,value):
        self.widthglue2 = value
        if self.start ==1:
            self.update_fun()
    def differentthreshold2box_fun(self,value):
        self.differentthreshold2 = value
        if self.start ==1:
            self.update_fun()
    def anabrightness2box_fun(self,value):
        self.anabrightness2 = value
        if self.start ==1:
            self.update_fun()
    def anacontrast2box_fun(self,value):
        self.anacontrast2 = value
        if self.start ==1:
            self.update_fun()
    def kerneldilate2box_fun(self,value):
        self.kerneldilate2= value
        if self.start ==1:
            self.update_fun()
    def kernelerode2box_fun(self,value):
        self.kernelerode2 = value
        if self.start ==1:
            self.update_fun()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera01frame.setText(_translate("MainWindow", "Camera number 01"))
        self.savedata1btn.setText(_translate("MainWindow", "SAVE DATA"))
        self.X_Point_W_9.setText(_translate("MainWindow", "<html><head/><body><p>Adjust brightness</p></body></html>"))
        self.Y_Point_H_3.setText(_translate("MainWindow", "<html><head/><body><p>Adjust lower frame</p></body></html>"))
        self.startcam1btn.setText(_translate("MainWindow", "START CAMERA 01"))
        self.X_Point_3.setText(_translate("MainWindow", "<html><head/><body><p>Adjust left frame</p></body></html>"))
        self.X_Point_W_10.setText(_translate("MainWindow", "<html><head/><body><p>Adjust contrast</p></body></html>"))
        self.savesample1btn.setText(_translate("MainWindow", "SAVE SAMPLE"))
        self.Y_Point_3.setText(_translate("MainWindow", "<html><head/><body><p>Adjust Upper frame</p></body></html>"))
        self.X_Point_W_3.setText(_translate("MainWindow", "<html><head/><body><p>Adjust right frame</p></body></html>"))
        self.Val_max_9.setText(_translate("MainWindow", "<html><head/><body><p>Contrast</p></body></html>"))
        self.Sat_min_3.setText(_translate("MainWindow", "<html><head/><body><p>Inner elip A</p></body></html>"))
        self.contour_number_7.setText(_translate("MainWindow", "<html><head/><body><p>Contours</p></body></html>"))
        self.anacam1frame.setText(_translate("MainWindow", "Analysis picture"))
        self.Val_max_10.setText(_translate("MainWindow", "<html><head/><body><p>Inner Dir</p></body></html>"))
        self.Hue_min_3.setText(_translate("MainWindow", "<html><head/><body><p>Outer elip A</p></body></html>"))
        self.AreaOK_3.setText(_translate("MainWindow", "<html><head/><body><p>Set different</p></body></html>"))
        self.Val_max_11.setText(_translate("MainWindow", "<html><head/><body><p>kernel dilate</p></body></html>"))
        self.Val_max_12.setText(_translate("MainWindow", "<html><head/><body><p>Width glue</p></body></html>"))
        self.numberconok_3.setText(_translate("MainWindow", "<html><head/><body><p>Adj different</p></body></html>"))
        self.contour_number_8.setText(_translate("MainWindow", "<html><head/><body><p>Set Contours</p></body></html>"))
        self.contour_number_9.setText(_translate("MainWindow", "<html><head/><body><p>Lengh min</p></body></html>"))
        self.thresh_ok_3.setText(_translate("MainWindow", "<html><head/><body><p>Set value</p></body></html>"))
        self.Val_min_3.setText(_translate("MainWindow", "<html><head/><body><p>Outer Dir</p></body></html>"))
        self.thresh_adj_3.setText(_translate("MainWindow", "<html><head/><body><p>Adj threshold</p></body></html>"))
        self.threshold_3.setText(_translate("MainWindow", "<html><head/><body><p>Actual pixel</p></body></html>"))
        self.contour_number_10.setText(_translate("MainWindow", "<html><head/><body><p>Lengh max</p></body></html>"))
        self.contour_number_11.setText(_translate("MainWindow", "<html><head/><body><p>Actual lengh</p></body></html>"))
        self.Val_max_13.setText(_translate("MainWindow", "<html><head/><body><p>kernel Erode</p></body></html>"))
        self.Sat_max_3.setText(_translate("MainWindow", "<html><head/><body><p>Inner elip B</p></body></html>"))
        self.Val_max_14.setText(_translate("MainWindow", "<html><head/><body><p>Brightness</p></body></html>"))
        self.contour_area_3.setText(_translate("MainWindow", "<html><head/><body><p>Actual different</p></body></html>"))
        self.Val_max_15.setText(_translate("MainWindow", "<html><head/><body><p>Different Threshold</p></body></html>"))
        self.Hue_max_3.setText(_translate("MainWindow", "<html><head/><body><p>Outer elip B</p></body></html>"))
        self.Y_Point_H_2.setText(_translate("MainWindow", "<html><head/><body><p>Adjust lower frame</p></body></html>"))
        self.savedata2btn.setText(_translate("MainWindow", "SAVE DATA"))
        self.Y_Point_2.setText(_translate("MainWindow", "<html><head/><body><p>Adjust Upper frame</p></body></html>"))
        self.X_Point_2.setText(_translate("MainWindow", "<html><head/><body><p>Adjust left frame</p></body></html>"))
        self.camera02frame.setText(_translate("MainWindow", "Camera number 02"))
        self.X_Point_W_2.setText(_translate("MainWindow", "<html><head/><body><p>Adjust right frame</p></body></html>"))
        self.X_Point_W_8.setText(_translate("MainWindow", "<html><head/><body><p>Adjust contrast</p></body></html>"))
        self.X_Point_W_7.setText(_translate("MainWindow", "<html><head/><body><p>Adjust brightness</p></body></html>"))
        self.startcam2btn.setText(_translate("MainWindow", "START CAMERA 02"))
        self.savesample2btn.setText(_translate("MainWindow", "SAVE SAMPLE"))
        self.thresh_ok_2.setText(_translate("MainWindow", "<html><head/><body><p>Set value</p></body></html>"))
        self.numberconok_2.setText(_translate("MainWindow", "<html><head/><body><p>Adj different</p></body></html>"))
        self.AreaOK_2.setText(_translate("MainWindow", "<html><head/><body><p>Set different</p></body></html>"))
        self.Sat_max_2.setText(_translate("MainWindow", "<html><head/><body><p>Inner elip B</p></body></html>"))
        self.Hue_min_2.setText(_translate("MainWindow", "<html><head/><body><p>Outer elip A</p></body></html>"))
        self.Val_min_2.setText(_translate("MainWindow", "<html><head/><body><p>Outer Dir</p></body></html>"))
        self.Val_max_2.setText(_translate("MainWindow", "<html><head/><body><p>Inner Dir</p></body></html>"))
        self.Hue_max_2.setText(_translate("MainWindow", "<html><head/><body><p>Outer elip B</p></body></html>"))
        self.contour_area_2.setText(_translate("MainWindow", "<html><head/><body><p>Actual different</p></body></html>"))
        self.anacam2frame.setText(_translate("MainWindow", "Analysis picture 2"))
        self.threshold_2.setText(_translate("MainWindow", "<html><head/><body><p>Actual pixel</p></body></html>"))
        self.contour_number_2.setText(_translate("MainWindow", "<html><head/><body><p>Contours</p></body></html>"))
        self.thresh_adj_2.setText(_translate("MainWindow", "<html><head/><body><p>Adj threshold</p></body></html>"))
        self.Sat_min_2.setText(_translate("MainWindow", "<html><head/><body><p>Inner elip A</p></body></html>"))
        self.contour_number_3.setText(_translate("MainWindow", "<html><head/><body><p>Set Contours</p></body></html>"))
        self.contour_number_4.setText(_translate("MainWindow", "<html><head/><body><p>Actual lengh</p></body></html>"))
        self.contour_number_5.setText(_translate("MainWindow", "<html><head/><body><p>Lengh min</p></body></html>"))
        self.contour_number_6.setText(_translate("MainWindow", "<html><head/><body><p>Lengh max</p></body></html>"))
        self.Val_max_3.setText(_translate("MainWindow", "<html><head/><body><p>Brightness</p></body></html>"))
        self.Val_max_4.setText(_translate("MainWindow", "<html><head/><body><p>Contrast</p></body></html>"))
        self.Val_max_5.setText(_translate("MainWindow", "<html><head/><body><p>kernel dilate</p></body></html>"))
        self.Val_max_6.setText(_translate("MainWindow", "<html><head/><body><p>Width glue</p></body></html>"))
        self.Val_max_7.setText(_translate("MainWindow", "<html><head/><body><p>Different Threshold</p></body></html>"))
        self.Val_max_8.setText(_translate("MainWindow", "<html><head/><body><p>kernel Erode</p></body></html>"))
        self.video1btn.setText(_translate("MainWindow", "Video 1"))
        self.anavideo2btn.setText(_translate("MainWindow", "Analyse  Video 2"))
        self.video2btn.setText(_translate("MainWindow", "Video 2"))
        self.anavideo1btn.setText(_translate("MainWindow", "Analyse video 1"))
        self.Model.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Running model</span></p></body></html>"))
        self.createdatabtn.setText(_translate("MainWindow", "Create Data"))
        self.Model_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Q\'ty cam 1</span></p></body></html>"))
        self.Model_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Q\'ty cam 2</span></p></body></html>"))
        self.Model_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Copy to model</span></p></body></html>"))
        self.copydatabtn.setText(_translate("MainWindow", "Copy model"))
        self.Model_2.setText(_translate("MainWindow", "<html><head/><body><p>Video ID</p></body></html>"))
        self.deletedatabtn.setText(_translate("MainWindow", "Delete data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

