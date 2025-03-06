import cv2
import numpy as np
import glob

# Kích thước bàn cờ (số ô vuông - 1)
chessboard_size = (7, 7)  # Số ô vuông theo chiều ngang và dọc
frame_size = None  # Kích thước ảnh sẽ được xác định tự động

# Tạo danh sách để lưu các điểm 3D và 2D
objpoints = []  # Điểm 3D trong không gian thực
imgpoints = []  # Điểm 2D trong ảnh

# Tạo các điểm 3D cho bàn cờ
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Đường dẫn đến thư mục chứa ảnh hiệu chỉnh
images = glob.glob('captured_images/*.jpg')
print(f"Số lượng ảnh tìm thấy: {len(images)}")
# Kiểm tra xem có ảnh nào được tìm thấy không
if len(images) == 0:
    print("Không tìm thấy ảnh trong thư mục. Hãy kiểm tra lại đường dẫn.")
    exit()

# Duyệt qua từng ảnh để tìm các góc của bàn cờ
for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Chessboard', gray)
    #key = cv2.waitKey(1)
    #if key == ord('n'):

    #Tìm các góc của bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(ret)
    #
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

         # Vẽ và hiển thị các góc
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Kiểm tra số lượng ảnh có điểm ảnh được phát hiện
if len(objpoints) == 0:
    print("Không tìm thấy bàn cờ trong bất kỳ ảnh nào. Hãy kiểm tra lại ảnh và kích thước bàn cờ.")
    exit()

# Hiệu chỉnh camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Khử méo ảnh
for image_path in images:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Tạo ma trận mới để loại bỏ méo
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Khử méo ảnh
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Cắt ảnh để loại bỏ các vùng đen do khử méo
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Hiển thị ảnh gốc và ảnh đã khử méo
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(500)

cv2.destroyAllWindows()
#import cv2
import numpy as np

# Ma trận camera và hệ số méo từ quá trình hiệu chỉnh
#camera_matrix = np.array([[fx, 0, cx],
                          #[0, fy, cy],
                          #[0, 0, 1]])  # Thay thế bằng giá trị thực tế của bạn
#dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Thay thế bằng giá trị thực tế của bạn

# Mở camera
cap = cv2.VideoCapture(1)  # Số 0 là camera mặc định, thay đổi nếu cần

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình.")
        break

    # Lấy kích thước khung hình
    h, w = frame.shape[:2]

    # Tạo ma trận mới để loại bỏ méo
    #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        #camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Khử méo khung hình
    #undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Cắt ảnh để loại bỏ các vùng đen do khử méo (nếu cần)
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Hiển thị khung hình gốc và khung hình đã khử méo
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Undistorted Frame', undistorted_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()