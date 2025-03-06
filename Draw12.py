import cv2
import numpy as np

# Biến toàn cục
shapes = []
drawing = False
dragging = False
resizing = False
type_selected = "rectangle"
selected_shape = None
selected_corner = None
offset_x = 0
offset_y = 0
resize_threshold = 10
donut_thickness = 10
polygon_points = []
current_img = None

def get_corner(shape, x, y):
    x1, y1, x2, y2 = shape["coords"]
    corners = {
        "tl": (x1, y1),
        "tr": (x2, y1),
        "bl": (x1, y2),
        "br": (x2, y2)
    }
    for key, (cx, cy) in corners.items():
        if abs(x - cx) < resize_threshold and abs(y - cy) < resize_threshold:
            return key
    return None

def mouse_callback(event, x, y, flags, param):
    global shapes, drawing, dragging, resizing, selected_shape, selected_corner, offset_x, offset_y, donut_thickness, polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if type_selected == "polygon":
            polygon_points.append((x, y))
            return

        for shape in shapes:
            # Chỉ xử lý các hình có "coords" (không phải đa giác)
            if shape["type"] != "polygon":
                corner = get_corner(shape, x, y)
                if corner:
                    resizing = True
                    selected_shape = shape
                    selected_corner = corner
                    return
                x1, y1, x2, y2 = shape["coords"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    dragging = True
                    selected_shape = shape
                    offset_x = x - x1
                    offset_y = y - y1
                    return

        drawing = True
        shapes.append({"type": type_selected, "coords": [x, y, x, y], "thickness": donut_thickness})

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and type_selected != "polygon":
            shapes[-1]["coords"][2] = x
            shapes[-1]["coords"][3] = y
        elif dragging and selected_shape:
            w = selected_shape["coords"][2] - selected_shape["coords"][0]
            h = selected_shape["coords"][3] - selected_shape["coords"][1]
            selected_shape["coords"] = [x - offset_x, y - offset_y, x - offset_x + w, y - offset_y + h]
        elif resizing and selected_shape:
            if selected_corner == "tl":
                selected_shape["coords"][0] = x
                selected_shape["coords"][1] = y
            elif selected_corner == "tr":
                selected_shape["coords"][2] = x
                selected_shape["coords"][1] = y
            elif selected_corner == "bl":
                selected_shape["coords"][0] = x
                selected_shape["coords"][3] = y
            elif selected_corner == "br":
                selected_shape["coords"][2] = x
                selected_shape["coords"][3] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if type_selected != "polygon":
            drawing = False
            dragging = False
            resizing = False
            selected_shape = None
            selected_corner = None

def save_cropped_images():
    global shapes, current_img
    for i, shape in enumerate(shapes):
        if shape["type"] == "polygon":
            points = np.array(shape["points"], np.int32)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1 = max(0, min(x_coords))
            y1 = max(0, min(y_coords))
            x2 = min(current_img.shape[1], max(x_coords))
            y2 = min(current_img.shape[0], max(y_coords))
            if x1 >= x2 or y1 >= y2:
                continue
            cropped_img = current_img[y1:y2, x1:x2].copy()
            mask = np.zeros_like(cropped_img)
            adjusted_points = [(p[0] - x1, p[1] - y1) for p in points]
            cv2.fillPoly(mask, [np.array(adjusted_points, np.int32)], (255, 255, 255))
            result = cv2.bitwise_and(cropped_img, mask)
            result[mask == 0] = 0
            cv2.imwrite(f"shape_{i}.png", result)
        else:
            x1, y1, x2, y2 = shape["coords"]
            cropped_img = current_img[y1:y2, x1:x2].copy()
            mask = np.zeros_like(cropped_img)
            if shape["type"] == "rectangle":
                mask[:] = 255
            elif shape["type"] == "circle":
                h, w = cropped_img.shape[:2]
                center = (w // 2, h // 2)
                radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                cv2.circle(mask, center, radius, (255, 255, 255), -1)
            elif shape["type"] == "ellipse":
                h, w = cropped_img.shape[:2]
                center = (w // 2, h // 2)
                axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
            elif shape["type"] == "donut":
                h, w = cropped_img.shape[:2]
                center = (w // 2, h // 2)
                outer_axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                inner_axes = (
                    max(1, outer_axes[0] - shape["thickness"]),
                    max(1, outer_axes[1] - shape["thickness"])
                )
                cv2.ellipse(mask, center, outer_axes, 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(mask, center, inner_axes, 0, 0, 360, (0, 0, 0), -1)
            result = cv2.bitwise_and(cropped_img, mask)
            result[mask == 0] = 0
            cv2.imwrite(f"shape_{i}.png", result)

# Tạo cửa sổ và gán callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Ảnh nền
img = np.ones((500, 800, 3), dtype=np.uint8) * 255

while True:
    temp_img = img.copy()

    # Vẽ tất cả các hình
    for shape in shapes:
        if shape["type"] == "polygon":
            points = np.array(shape["points"], np.int32)
            cv2.polylines(temp_img, [points], True, (255, 0, 255), 2)
        else:
            x1, y1, x2, y2 = shape["coords"]
            if shape["type"] == "rectangle":
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif shape["type"] == "circle":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                cv2.circle(temp_img, center, radius, (255, 0, 0), 2)
            elif shape["type"] == "ellipse":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                cv2.ellipse(temp_img, center, axes, 0, 0, 360, (0, 0, 255), 2)
            elif shape["type"] == "donut":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                inner_axes = (max(1, axes[0] - shape["thickness"]), max(1, axes[1] - shape["thickness"]))
                cv2.ellipse(temp_img, center, axes, 0, 0, 360, (0, 255, 255), 2)
                cv2.ellipse(temp_img, center, inner_axes, 0, 0, 360, (0, 255, 0), 2)

    # Vẽ đa giác đang được vẽ
    if type_selected == "polygon" and len(polygon_points) > 0:
        for i in range(len(polygon_points) - 1):
            cv2.line(temp_img, polygon_points[i], polygon_points[i + 1], (255, 0, 255), 2)
        if len(polygon_points) > 1:
            cv2.line(temp_img, polygon_points[-1], polygon_points[0], (255, 0, 255), 2)

    cv2.imshow("Image", temp_img)
    current_img = temp_img.copy()
    key = cv2.waitKey(1)

    if key == ord('r'):
        type_selected = "rectangle"
    elif key == ord('c'):
        type_selected = "circle"
    elif key == ord('e'):
        type_selected = "ellipse"
    elif key == ord('d'):
        type_selected = "donut"
    elif key == ord('p'):
        type_selected = "polygon"
    elif key == ord('+'):
        donut_thickness += 2
    elif key == ord('-'):
        donut_thickness = max(2, donut_thickness - 2)

    if key in [32, 0] and shapes:
        shapes.pop()
    elif key == ord('s'):
        save_cropped_images()
    elif key == 13:  # Phím Enter
        if type_selected == "polygon" and len(polygon_points) >= 3:
            shapes.append({"type": "polygon", "points": polygon_points.copy()})
            polygon_points.clear()
    elif key == 27:  # Phím ESC
        break

cv2.destroyAllWindows()