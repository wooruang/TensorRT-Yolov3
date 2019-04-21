import sys
import cv2
import time

sys.path.insert(0, "cmake-build-debug")

import libTensorRTYolov3

a = libTensorRTYolov3.Yolov3TensorRT("cmake-build-debug/yolov3_fp32.engine", 608, 608, 3, 0.4, 80)

print(a)

img = cv2.imread("cmake-build-debug/test.jpg")

print(img)

a.init()

while True:
    s = time.time()
    print( img.shape)
    h, w, c = img.shape
    print(h, w, c)
    c = len(img.shape)

    b = a.predictFromPython(img, w, h, c)

    print(b)
    e = time.time()
    print("Total : %s" % (e - s))

