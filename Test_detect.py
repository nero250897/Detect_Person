from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')
results = model('C:/Users/84335/PycharmProjects/Yolov8/images/person4.jpg', show=True)

cv2.waitKey(0)


