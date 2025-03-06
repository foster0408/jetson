from ultralytics import YOLO

# Load mô hình từ file .engine
#model = YOLO("runs/detect/train4/weights/best.engine", task="detect")
model = YOLO("yolov8n.engine", task="detect")

# Đọc ảnh
import cv2

cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame từ webcam
    if not ret:
        break
    
    results = model.predict(frame, imgsz=640)  # Chạy YOLO
    annotated_frame = results[0].plot()  # Vẽ kết quả

    cv2.imshow("YOLO Detection", annotated_frame)  # Hiển thị

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()


