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
#from cv2 import imshow
#from threading import Thread, local
import nanocamera as nano
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
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2

# Function to load TensorRT engine
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Function to allocate memory and prepare buffers
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    return h_input, h_output, d_input, d_output, bindings, stream

# Function to perform inference
def infer(engine, context, bindings, stream, h_input, h_output, d_input, d_output):
    # Transfer input data to GPU
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer output data to CPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

# Load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(224, 224), swapRB=True, crop=False)
    return blob, h, w

# Post-process output (e.g., for classification)
def postprocess_output(output, h, w):
    # Example: assume output is classification probabilities
    class_id = np.argmax(output)
    confidence = output[0][class_id]
    return class_id, confidence

def main():
   if True:
    
        engine_path = 'model.trt'
        image_path = 'image.jpg'

        # Load TensorRT engine
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # Allocate buffers
        h_input, h_output, d_input, d_output, bindings, stream = allocate_buffers(engine)

        # Preprocess image
        blob, h, w = preprocess_image(image_path)
        np.copyto(h_input, blob.ravel())

        # Perform inference
        output = infer(engine, context, bindings, stream, h_input, h_output, d_input, d_output)
        print(output)
        # Post-process output
        #class_id, confidence = postprocess_output(output, h, w)
        #print(f'Class ID: {class_id}, Confidence: {confidence}')

if __name__ == '__main__':
    main()

