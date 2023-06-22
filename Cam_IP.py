from ultralytics import YOLO
import cv2
from imutils.video import VideoStream
import math
import numpy as np
import pyglet

# Kích thước khung hình
#frame_width = 640
#frame_height = 480

# lay anh tu duong truyen cam ip
url = "rtsp://admin:L2AC931D@192.168.1.99:554/cam/realmonitor?channel=1&subtype=00"
video = VideoStream(url).start()


# Su dung yolov8
model = YOLO("best.pt")
classNames = ["person"]


# Ve pham vi quan sat doi tuong di chuyen vao
top_left, bottom_right = (350, 5), (200, 300)

while True:
    img = video.read()
    # resize video xuat tu cam ip
    results = model(img, stream=True)
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 255), 2)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Tinh toan centroid
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            cv2.circle(img, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            # print(box.conf[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            # print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [0, 255, 0], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            # Kiểm tra đối tượng có nằm trong khu vực quan sát hay không
            logic = top_left[0] < centroid_x < bottom_right[0] and top_left[1] < centroid_y < bottom_right[1]
            if logic:
                text = "Co xam nhap"
                # Hiện cảnh báo lên hình
                cv2.putText(img, "Canh bao: {}".format(text), (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
out.release()
cv2.destroyAllWindows()