import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture("C:/Users/84335/PycharmProjects/Yolov8/people.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for result in results:
            boxes = result.boxes.numpy()
            print("boxes", boxes)
            for box in boxes:
                print("class", box.cls)
                print("xyxy", box.xyxy)
                print("conf", box.conf)
    else:
        break

cap.release()
cv2.destroyAllWindows()
