import cv2

# Đọc ảnh
image = cv2.imread('C:/Users/84335/PycharmProjects/Yolov8/Detect_Person/resize_frame.jpg')

# Vẽ khung
top_left = (350, 5)
bottom_right = (200, 300)
frame_image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Hiển thị ảnh với khung
cv2.imshow('Frame Image', frame_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
