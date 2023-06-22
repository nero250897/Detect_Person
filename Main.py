from ultralytics import YOLO
import cv2
import math
import numpy as np
import time
#import threading

cap = cv2.VideoCapture("C:/Users/84335/PycharmProjects/Doan/Main3.mp4")

# Mang chua cac diem de ve da giac
polygons = [
    np.array([[248, 278],[488, 270],[552, 386],[160, 386],[236, 290]], np.int32)
    ]

model = YOLO("best.pt")
classNames = ["person"]

# Khoi tao bien de tinh toan FPS
start_time = time.time()
frame_count = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Ve cac da giac len anh
    for polygon in polygons:
        cv2.polylines(img, [polygon], isClosed=True, color=(0, 0, 255), thickness=3) # Mau do

    for r in results: # Vong lap hien thi bouding box cho moi ket qua
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3) #Mau xanh

            # Tinh toan centroid
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(img, centroid, 5, (0, 255, 0), -1) #Mau xanh la
            centroid_array = np.array(centroid)
            #print(centroid_array)

            # print(box.conf[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

            # print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)  # filled Mau xanh la
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA) #Mau trang

    # Kiem tra doi tuong co trong khu vuc polygon hay khong
    is_inside = cv2.pointPolygonTest(polygon, tuple(centroid), False)

    if is_inside >= 0:
        text = "Co xam nhap"
        # Hien canh bao len man hinh
        cv2.putText(img, "Canh bao: {}".format(text), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # In anh
        cv2.imwrite("alert.png", img) #cv2.resize(img, dsize=None, fx=0.2, fy=0.2

    # Tinh toan FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    resized_frame = cv2.resize(img, (960, 540))
    cv2.imshow("Image", resized_frame)

cap.release()
cv2.destroyAllWindows()


