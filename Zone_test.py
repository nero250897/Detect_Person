import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

def draw_rectangle(frame, xyxy, names, cls, cof):

    # Hien thi kich thuoc anh
    #h, w, c = frame.shape
    #print("...", h, w)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (int(xyxy[0][0]), int(xyxy[0][1]))
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    if int(xyxy[0][1]) < 200:
    #if int(xyxy[0][1]) < int(frame.shape[0]/2):
        frame = cv2.putText(frame, names[cls[0]] + " " + str(round(cof[0],2)),
                            org, font, fontScale, color, thickness, cv2.LINE_AA)

    start_point = (500, 500)

    end_point = (100, 400)

    color = (255, 0, 0)

    thickness = 2

    image = cv2.rectangle(frame, start_point, end_point, color, thickness)

    scale_percent = 90 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize anh
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Zone", resized)


cap = cv2.VideoCapture("C:/Users/84335/PycharmProjects/Yolov8/people.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for result in results:
            boxes = result.boxes.numpy()
            names = result.names
            print("boxes", boxes)
            for box in boxes:
                print("class", box.cls)
                print("xyxy", box.xyxy)
                print("conf", box.conf)
                draw_rectangle(frame, box.xyxy, names, box.cls, box.conf)
    else:
        break

cap.release()
cv2.destroyAllWindows()
