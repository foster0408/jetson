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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Khởi tạo logger của TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Hàm để tải mô hình TensorRT
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Hàm để thực hiện inferencing trên ảnh đầu vào
def do_inference(context, bindings, inputs, outputs, stream):
    # Copy dữ liệu đầu vào từ host sang device
    [cuda.memcpy_htod_async(inp, inp_host, stream) for inp, inp_host in zip(bindings, inputs)]
    # Thực hiện inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Copy kết quả từ device sang host
    [cuda.memcpy_dtoh_async(out_host, out, stream) for out, out_host in zip(outputs, outputs)]
    # Đợi mọi thứ hoàn tất
    stream.synchronize()

# Chuẩn bị ảnh đầu vào
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, input_shape)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # CHW
    return np.expand_dims(image_transposed, axis=0)

# Hàm để post-process kết quả trả về từ mô hình
def postprocess_results(output, img_shape, conf_thresh=0.5):
    # Xử lý đầu ra từ TensorRT để lấy bounding box và lớp dự đoán
    # Đây chỉ là một ví dụ đơn giản, thực tế có thể phức tạp hơn
    boxes, classes, confidences = [], [], []
    for detection in output[0]:
        confidence = float(detection[4])
        if confidence > conf_thresh:
            x, y, w, h = detection[:4]
            boxes.append([x, y, w, h])
            confidences.append(confidence)
            classes.append(int(detection[5]))
    return boxes, classes, confidences

# Hàm để vẽ bounding boxes lên ảnh
def draw_boxes(image, boxes, confidences, classes, class_names):
    for box, conf, cls in zip(boxes, confidences, classes):
        x, y, w, h = box
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.putText(image, label, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# Đường dẫn đến mô hình TensorRT và ảnh đầu vào
engine_file_path = "model.trt"
image_path = "input.jpg"
output_image_path = "output.jpg"

# Tải mô hình TensorRT
engine = load_engine(engine_file_path)
context = engine.create_execution_context()

# Chuẩn bị dữ liệu cho inference
input_shape = (640, 640)  # Kích thước đầu vào của mô hình
input_image = preprocess_image(image_path, input_shape)

# Chuẩn bị các buffer cho TensorRT
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Tạo các buffer cho đầu vào và đầu ra
    if engine.binding_is_input(binding):
        inputs.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
        inputs_host = np.ascontiguousarray(input_image)
    else:
        outputs.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
        outputs_host = np.empty(size, dtype=dtype)
    bindings.append(int(inputs[-1]) if engine.binding_is_input(binding) else int(outputs[-1]))

# Thực hiện inference
do_inference(context, bindings, [inputs_host], [outputs_host], stream)

# Post-process kết quả
boxes, classes, confidences = postprocess_results(outputs_host, input_shape)

# Tải ảnh gốc và vẽ kết quả
image = cv2.imread(image_path)
class_names = ["class1", "class2", "class3"]  # Danh sách tên các lớp
output_image = draw_boxes(image, boxes, confidences, classes, class_names)

# Lưu ảnh kết quả
cv2.imwrite(output_image_path, output_image)

