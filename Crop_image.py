import cv2
import numpy as np

# Khởi tạo biến toàn cục
drawing = False
current_shape = "rectangle"  # rectangle|circle|ellipse|annulus|polygon
image = None
clone = None
shapes = []  # Lưu trữ tất cả các hình đã vẽ
temp_points = []  # Điểm tạm thời khi vẽ
current_mask = None  # Mask hiện tại

# Các tham số hình dạng
center = (0, 0)
axes = (0, 0)  # Cho ellipse
radius = 0
radius_inner = 0  # Cho vành khuyên


def draw_shape(event, x, y, flags, param):
    global drawing, clone, image, temp_points, center, radius, axes, radius_inner

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp_points = [(x, y)]

        if current_shape == "rectangle":
            temp_points.append((x, y))
        elif current_shape in ["circle", "annulus"]:
            center = (x, y)
        elif current_shape == "ellipse":
            center = (x, y)
        elif current_shape == "polygon":
            temp_points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if current_shape == "rectangle":
            clone = image.copy()
            cv2.rectangle(clone, temp_points[0], (x, y), (0, 255, 0), 2)
        elif current_shape == "circle":
            clone = image.copy()
            radius = int(np.hypot(x - center[0], y - center[1]))
            cv2.circle(clone, center, radius, (0, 255, 0), 2)
        elif current_shape == "annulus":
            clone = image.copy()
            radius = int(np.hypot(x - center[0], y - center[1]))
            cv2.circle(clone, center, radius, (0, 255, 0), 2)
            cv2.circle(clone, center, radius_inner, (0, 255, 0), 2)
        elif current_shape == "ellipse":
            clone = image.copy()
            axes = (int(abs(x - center[0])), int(abs(y - center[1])))
            cv2.ellipse(clone, center, axes, 0, 0, 360, (0, 255, 0), 2)
        elif current_shape == "polygon":
            clone = image.copy()
            pts = np.array(temp_points + [(x, y)], np.int32)
            cv2.polylines(clone, [pts], False, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        if current_shape == "rectangle":
            shapes.append(("rectangle", temp_points[0], (x, y)))
        elif current_shape == "circle":
            shapes.append(("circle", center, radius))
        elif current_shape == "annulus":
            radius_inner = int(input("Nhập bán kính trong: "))
            shapes.append(("annulus", center, radius, radius_inner))
        elif current_shape == "ellipse":
            shapes.append(("ellipse", center, axes))
        elif current_shape == "polygon":
            shapes.append(("polygon", temp_points))

        process_image()
        temp_points = []


def process_image():
    global image, current_mask
    if not shapes: return

    # Tạo mask tổng hợp
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for shape in shapes:
        if shape[0] == "rectangle":
            cv2.rectangle(mask, shape[1], shape[2], 255, -1)
        elif shape[0] == "circle":
            cv2.circle(mask, shape[1], shape[2], 255, -1)
        elif shape[0] == "annulus":
            cv2.circle(mask, shape[1], shape[3], 0, -1)  # Inner circle
            cv2.circle(mask, shape[1], shape[2], 255, -1)
        elif shape[0] == "ellipse":
            cv2.ellipse(mask, shape[1], shape[2], 0, 0, 360, 255, -1)
        elif shape[0] == "polygon":
            pts = np.array(shape[1], np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # Áp dụng xử lý ảnh (VD: Làm mờ)
    processed = cv2.bitwise_and(image, image, mask=mask)
    blurred = cv2.GaussianBlur(processed, (1, 1), 0)
    image = cv2.add(image, blurred, mask=mask)
    current_mask = mask.copy()
    cv2.imshow("Image", image)


# Khởi tạo ảnh
image = cv2.imread('input.jpg')
clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_shape)

while True:
    cv2.imshow("Image", clone)
    key = cv2.waitKey(1) & 0xFF

    # Chọn hình dạng
    if key == ord('r'):
        current_shape = "rectangle"
    elif key == ord('c'):
        current_shape = "circle"
    elif key == ord('a'):
        current_shape = "annulus"
    elif key == ord('e'):
        current_shape = "ellipse"
    elif key == ord('p'):
        current_shape = "polygon"
    elif key == ord('s'):  # Lưu mask
        if current_mask is not None:
            cv2.imwrite("mask.png", current_mask)
    elif key == 27:
        break  # ESC để thoát

cv2.destroyAllWindows()