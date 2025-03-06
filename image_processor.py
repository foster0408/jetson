import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        # Biến toàn cục
        self.shapes = []
        self.drawing = False
        self.dragging = False
        self.resizing = False
        self.type_selected = "rectangle"
        self.selected_shape = None
        self.selected_corner = None
        self.offset_x = 0
        self.offset_y = 0
        self.resize_threshold = 10
        self.donut_thickness = 10
        self.polygon_points = []
        self.current_img = None
        self.rotating = False  # Trạng thái xoay
        self.scale = 1.0  # Tỉ lệ phóng to/thu nhỏ

        # Mở camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Không thể mở camera.")

    def get_corner(self, shape, x, y):
        x1, y1, x2, y2 = shape["coords"]
        corners = {
            "tl": (x1, y1),
            "tr": (x2, y1),
            "bl": (x1, y2),
            "br": (x2, y2)
        }
        for key, (cx, cy) in corners.items():
            if abs(x - cx) < self.resize_threshold and abs(y - cy) < self.resize_threshold:
                return key
        return None

    def mouse_callback(self, event, x, y, flags, param):
        # Điều chỉnh tọa độ chuột dựa trên tỉ lệ phóng to/thu nhỏ
        x = int(x / self.scale)
        y = int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.type_selected == "polygon":
                self.polygon_points.append((x, y))
                return

            for shape in self.shapes:
                if shape["type"] != "polygon":
                    corner = self.get_corner(shape, x, y)
                    if corner:
                        self.resizing = True
                        self.selected_shape = shape
                        self.selected_corner = corner
                        return
                    x1, y1, x2, y2 = shape["coords"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.dragging = True
                        self.selected_shape = shape
                        self.offset_x = x - x1
                        self.offset_y = y - y1
                        return

            self.drawing = True
            self.shapes.append({"type": self.type_selected, "coords": [x, y, x, y], "thickness": self.donut_thickness, "angle": 0})

        elif event == cv2.EVENT_RBUTTONDOWN:  # Nhấn chuột phải để xoay
            for shape in self.shapes:
                if shape["type"] in ["ellipse", "donut"]:
                    x1, y1, x2, y2 = shape["coords"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.rotating = True
                        self.selected_shape = shape
                        # Tính góc xoay dựa trên vị trí chuột
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        dx = x - center[0]
                        dy = y - center[1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        self.selected_shape["angle"] = angle
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.type_selected != "polygon":
                self.shapes[-1]["coords"][2] = x
                self.shapes[-1]["coords"][3] = y
            elif self.dragging and self.selected_shape:
                w = self.selected_shape["coords"][2] - self.selected_shape["coords"][0]
                h = self.selected_shape["coords"][3] - self.selected_shape["coords"][1]
                self.selected_shape["coords"] = [x - self.offset_x, y - self.offset_y, x - self.offset_x + w, y - self.offset_y + h]
            elif self.resizing and self.selected_shape:
                if self.selected_corner == "tl":
                    self.selected_shape["coords"][0] = x
                    self.selected_shape["coords"][1] = y
                elif self.selected_corner == "tr":
                    self.selected_shape["coords"][2] = x
                    self.selected_shape["coords"][1] = y
                elif self.selected_corner == "bl":
                    self.selected_shape["coords"][0] = x
                    self.selected_shape["coords"][3] = y
                elif self.selected_corner == "br":
                    self.selected_shape["coords"][2] = x
                    self.selected_shape["coords"][3] = y
            elif self.rotating and self.selected_shape:
                center = ((self.selected_shape["coords"][0] + self.selected_shape["coords"][2]) // 2,
                          (self.selected_shape["coords"][1] + self.selected_shape["coords"][3]) // 2)
                dx = x - center[0]
                dy = y - center[1]
                angle = np.degrees(np.arctan2(dy, dx))
                self.selected_shape["angle"] = angle

        elif event == cv2.EVENT_LBUTTONUP:
            if self.type_selected != "polygon":
                self.drawing = False
                self.dragging = False
                self.resizing = False
                self.rotating = False
                self.selected_shape = None
                self.selected_corner = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Lăn chuột lên
                self.scale *= 1.1
            else:  # Lăn chuột xuống
                self.scale /= 1.1
            self.scale = max(0.1, min(self.scale, 5.0))  # Giới hạn tỉ lệ phóng to/thu nhỏ

    def save_cropped_images(self):
        cropped_images = []  # Danh sách chứa các ảnh đã cắt
        for i, shape in enumerate(self.shapes):
            if shape["type"] == "polygon":
                points = np.array(shape["points"], np.int32)
                points = (points * self.scale).astype(np.int32)  # Điều chỉnh tọa độ theo tỉ lệ phóng to/thu nhỏ
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1 = max(0, min(x_coords))
                y1 = max(0, min(y_coords))
                x2 = min(self.current_img.shape[1], max(x_coords))
                y2 = min(self.current_img.shape[0], max(y_coords))
                if x1 >= x2 or y1 >= y2:
                    continue
                cropped_img = self.current_img[y1:y2, x1:x2].copy()
                mask = np.zeros_like(cropped_img)
                adjusted_points = [(p[0] - x1, p[1] - y1) for p in points]
                cv2.fillPoly(mask, [np.array(adjusted_points, np.int32)], (255, 255, 255))
                result = cv2.bitwise_and(cropped_img, mask)
                result[mask == 0] = 0
                cropped_images.append(result)  # Thêm ảnh đã cắt vào danh sách
            else:
                x1, y1, x2, y2 = shape["coords"]
                # Điều chỉnh tọa độ theo tỉ lệ phóng to/thu nhỏ
                x1, y1, x2, y2 = int(x1 * self.scale), int(y1 * self.scale), int(x2 * self.scale), int(y2 * self.scale)
                if shape["type"] in ["ellipse", "donut"]:
                    # Tính bounding box của hình elip (bao gồm cả góc xoay)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                    angle = shape["angle"]

                    # Tính toán bounding box dựa trên góc xoay
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    cos = np.abs(rotation_matrix[0, 0])
                    sin = np.abs(rotation_matrix[0, 1])
                    new_w = int((axes[0] * cos) + (axes[1] * sin))
                    new_h = int((axes[0] * sin) + (axes[1] * cos))

                    # Tạo mask cho ellipse đã xoay
                    mask = np.zeros_like(self.current_img)
                    if shape["type"] == "ellipse":
                        cv2.ellipse(mask, center, axes, angle, 0, 360, (255, 255, 255), -1)
                    elif shape["type"] == "donut":
                        outer_axes = axes
                        inner_axes = (
                            max(1, outer_axes[0] - shape["thickness"]),
                            max(1, outer_axes[1] - shape["thickness"])
                        )
                        cv2.ellipse(mask, center, outer_axes, angle, 0, 360, (255, 255, 255), -1)
                        cv2.ellipse(mask, center, inner_axes, angle, 0, 360, (0, 0, 0), -1)

                    # Áp dụng mask lên ảnh gốc
                    result = cv2.bitwise_and(self.current_img, mask)
                    result[mask == 0] = 0

                    # Cắt ảnh theo bounding box của hình elip
                    x1 = max(0, center[0] - new_w)
                    y1 = max(0, center[1] - new_h)
                    x2 = min(self.current_img.shape[1], center[0] + new_w)
                    y2 = min(self.current_img.shape[0], center[1] + new_h)
                    cropped_img = result[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)  # Thêm ảnh đã cắt vào danh sách
                else:
                    cropped_img = self.current_img[y1:y2, x1:x2].copy()
                    mask = np.zeros_like(cropped_img)
                    if shape["type"] == "rectangle":
                        mask[:] = 255
                    elif shape["type"] == "circle":
                        h, w = cropped_img.shape[:2]
                        center = (w // 2, h // 2)
                        radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                        cv2.circle(mask, center, radius, (255, 255, 255), -1)
                    result = cv2.bitwise_and(cropped_img, mask)
                    result[mask == 0] = 0
                    cropped_images.append(result)  # Thêm ảnh đã cắt vào danh sách
        return cropped_images  # Trả về danh sách các ảnh đã cắt

    def process(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Phóng to/thu nhỏ ảnh từ camera
            h, w = frame.shape[:2]
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))

            temp_img = resized_frame.copy()

            # Vẽ tất cả các hình
            for shape in self.shapes:
                if shape["type"] == "polygon":
                    points = np.array(shape["points"], np.int32)
                    points = (points * self.scale).astype(np.int32)  # Điều chỉnh tọa độ theo tỉ lệ
                    cv2.polylines(temp_img, [points], True, (255, 0, 255), 2)
                else:
                    x1, y1, x2, y2 = shape["coords"]
                    x1, y1, x2, y2 = int(x1 * self.scale), int(y1 * self.scale), int(x2 * self.scale), int(y2 * self.scale)  # Điều chỉnh tọa độ theo tỉ lệ
                    if shape["type"] == "rectangle":
                        cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif shape["type"] == "circle":
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                        cv2.circle(temp_img, center, radius, (255, 0, 0), 2)
                    elif shape["type"] == "ellipse":
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                        cv2.ellipse(temp_img, center, axes, shape["angle"], 0, 360, (0, 0, 255), 2)
                    elif shape["type"] == "donut":
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                        inner_axes = (max(1, axes[0] - shape["thickness"]), max(1, axes[1] - shape["thickness"]))
                        cv2.ellipse(temp_img, center, axes, shape["angle"], 0, 360, (0, 255, 255), 2)
                        cv2.ellipse(temp_img, center, inner_axes, shape["angle"], 0, 360, (0, 255, 0), 2)

            # Vẽ đa giác đang được vẽ
            if self.type_selected == "polygon" and len(self.polygon_points) > 0:
                for i in range(len(self.polygon_points) - 1):
                    pt1 = (int(self.polygon_points[i][0] * self.scale), int(self.polygon_points[i][1] * self.scale))
                    pt2 = (int(self.polygon_points[i + 1][0] * self.scale), int(self.polygon_points[i + 1][1] * self.scale))
                    cv2.line(temp_img, pt1, pt2, (255, 0, 255), 2)
                if len(self.polygon_points) > 1:
                    pt1 = (int(self.polygon_points[-1][0] * self.scale), int(self.polygon_points[-1][1] * self.scale))
                    pt2 = (int(self.polygon_points[0][0] * self.scale), int(self.polygon_points[0][1] * self.scale))
                    cv2.line(temp_img, pt1, pt2, (255, 0, 255), 2)

            cv2.imshow("Image", temp_img)
            self.current_img = resized_frame.copy()  # Lưu ảnh gốc đã phóng to/thu nhỏ
            key = cv2.waitKey(1)

            if key == ord('r'):
                self.type_selected = "rectangle"
            elif key == ord('c'):
                self.type_selected = "circle"
            elif key == ord('e'):
                self.type_selected = "ellipse"
            elif key == ord('d'):
                self.type_selected = "donut"
            elif key == ord('p'):
                self.type_selected = "polygon"
            elif key == ord('+'):
                self.donut_thickness += 2
            elif key == ord('-'):
                self.donut_thickness = max(2, self.donut_thickness - 2)

            if key in [32, 0] and self.shapes:
                self.shapes.pop()
            elif key == ord('s'):
                cropped_images = self.save_cropped_images()  # Lấy danh sách ảnh đã cắt
                return cropped_images  # Trả về danh sách ảnh đã cắt
            elif key == 13:  # Phím Enter
                if self.type_selected == "polygon" and len(self.polygon_points) >= 3:
                    self.shapes.append({"type": "polygon", "points": self.polygon_points.copy()})
                    self.polygon_points.clear()
            elif key == 27:  # Phím ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return []  # Trả về danh sách rỗng nếu không có ảnh nào được cắt