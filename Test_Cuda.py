import cv2
from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("yolov8n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame từ webcam
    if not ret:
        break
    
    results = model(frame)  # Chạy YOLO
    annotated_frame = results[0].plot()  # Vẽ kết quả

    cv2.imshow("YOLO Detection", annotated_frame)  # Hiển thị

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()


