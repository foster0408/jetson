import cv2
import os

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_folder = 'captured_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Mở camera
cap = cv2.VideoCapture(1)  # Số 0 là camera mặc định, thay đổi nếu cần

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Biến đếm số lượng ảnh đã chụp
image_count = 0

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình.")
        break

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    # Nhấn phím Enter (ASCII code 13) để chụp ảnh
    key = cv2.waitKey(1)
    if key == 13:  # Phím Enter
        # Tạo tên file ảnh
        image_name = os.path.join(output_folder, f'image_{image_count}.jpg')
        # Lưu ảnh
        cv2.imwrite(image_name, frame)
        print(f"Đã lưu ảnh: {image_name}")
        image_count += 1

    # Nhấn phím 'q' để thoát
    if key == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()