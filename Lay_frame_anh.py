from numpy import extract
from ultralytics import YOLO
import cv2
import supervision as sv


# Su dung yolov8
model = YOLO("best.pt")
classNames = ["person"]

VIDEO = "C:/Users/84335/PycharmProjects/Doan/Main1.mp4"

video_info = sv.VideoInfo.from_video_path(VIDEO)
colors = sv.ColorPalette.default()

#extract video frame
generator = sv.get_video_frames_generator(VIDEO)
iterator = iter(generator)

frame = next(iterator)

#save first frame
cv2.imwrite("first_frame.png", frame)
