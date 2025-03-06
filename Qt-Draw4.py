import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPolygon, QColor
from PyQt5.QtCore import Qt, QPoint, QTimer
import cv2

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
rotating = False  # Trạng thái xoay
scale_factor = 1.0  # Tỷ lệ phóng to/thu nhỏ

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

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.original_pixmap = None
        self.setPixmap(QPixmap())

    def mousePressEvent(self, event):
        global shapes, drawing, dragging, resizing, selected_shape, selected_corner, offset_x, offset_y, donut_thickness, polygon_points, rotating

        x, y = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            if type_selected == "polygon":
                polygon_points.append((x, y))
                self.update()
                return

            for shape in shapes:
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
            shapes.append({"type": type_selected, "coords": [x, y, x, y], "thickness": donut_thickness, "angle": 0})

        elif event.button() == Qt.RightButton:  # Nhấn chuột phải để xoay
            for shape in shapes:
                if shape["type"] in ["ellipse", "donut"]:
                    x1, y1, x2, y2 = shape["coords"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        rotating = True
                        selected_shape = shape
                        # Tính góc xoay dựa trên vị trí chuột
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        dx = x - center[0]
                        dy = y - center[1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        selected_shape["angle"] = angle
                        break

        self.update()

    def mouseMoveEvent(self, event):
        global shapes, drawing, dragging, resizing, selected_shape, selected_corner, offset_x, offset_y, donut_thickness, polygon_points, rotating

        x, y = event.x(), event.y()

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
        elif rotating and selected_shape:
            center = ((selected_shape["coords"][0] + selected_shape["coords"][2]) // 2,
                      (selected_shape["coords"][1] + selected_shape["coords"][3]) // 2)
            dx = x - center[0]
            dy = y - center[1]
            angle = np.degrees(np.arctan2(dy, dx))
            selected_shape["angle"] = angle

        self.update()

    def mouseReleaseEvent(self, event):
        global drawing, dragging, resizing, rotating, selected_shape, selected_corner

        if event.button() == Qt.LeftButton and type_selected != "polygon":
            drawing = False
            dragging = False
            resizing = False
            rotating = False
            selected_shape = None
            selected_corner = None

        self.update()

    def wheelEvent(self, event):
        global scale_factor
        delta = event.angleDelta().y()
        if delta > 0:
            scale_factor *= 1.1  # Phóng to
        else:
            scale_factor *= 0.9  # Thu nhỏ
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * scale_factor),
                int(self.original_pixmap.height() * scale_factor),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)

        for shape in shapes:
            if shape["type"] == "polygon":
                points = [QPoint(x, y) for x, y in shape["points"]]
                polygon = QPolygon(points)
                painter.drawPolygon(polygon)
            else:
                x1, y1, x2, y2 = shape["coords"]
                if shape["type"] == "rectangle":
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                elif shape["type"] == "circle":
                    center = QPoint((x1 + x2) // 2, (y1 + y2) // 2)
                    radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                    painter.drawEllipse(center, radius, radius)
                elif shape["type"] == "ellipse":
                    center = QPoint((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                    painter.save()
                    painter.translate(center)
                    painter.rotate(shape["angle"])
                    painter.translate(-center)
                    painter.drawEllipse(center, axes[0], axes[1])
                    painter.restore()
                elif shape["type"] == "donut":
                    center = QPoint((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                    inner_axes = (max(1, axes[0] - shape["thickness"]), max(1, axes[1] - shape["thickness"]))
                    painter.save()
                    painter.translate(center)
                    painter.rotate(shape["angle"])
                    painter.translate(-center)
                    painter.drawEllipse(center, axes[0], axes[1])
                    painter.drawEllipse(center, inner_axes[0], inner_axes[1])
                    painter.restore()

        if type_selected == "polygon" and len(polygon_points) > 0:
            for i in range(len(polygon_points) - 1):
                painter.drawLine(QPoint(*polygon_points[i]), QPoint(*polygon_points[i + 1]))
            if len(polygon_points) > 1:
                painter.drawLine(QPoint(*polygon_points[-1]), QPoint(*polygon_points[0]))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Drawing")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở camera")
            sys.exit()

        # Timer để cập nhật frame từ camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cập nhật frame mỗi 30ms

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.original_pixmap = QPixmap.fromImage(q_img)
            self.image_label.update_pixmap()

    def keyPressEvent(self, event):
        global type_selected, donut_thickness, shapes, polygon_points

        key = event.key()
        if key == Qt.Key_R:
            type_selected = "rectangle"
        elif key == Qt.Key_C:
            type_selected = "circle"
        elif key == Qt.Key_E:
            type_selected = "ellipse"
        elif key == Qt.Key_D:
            type_selected = "donut"
        elif key == Qt.Key_P:
            type_selected = "polygon"
        elif key == Qt.Key_Plus:
            donut_thickness += 2
        elif key == Qt.Key_Minus:
            donut_thickness = max(2, donut_thickness - 2)
        elif key == Qt.Key_Space or key == Qt.Key_0:
            if shapes:
                shapes.pop()
        elif key == Qt.Key_S:
            self.save_cropped_images()
        elif key == Qt.Key_Return:
            if type_selected == "polygon" and len(polygon_points) >= 3:
                shapes.append({"type": "polygon", "points": polygon_points.copy()})
                polygon_points.clear()
        elif key == Qt.Key_Escape:
            self.close()

        self.image_label.update()

    def save_cropped_images(self):
        global shapes, current_img

        # Lấy hình ảnh hiện tại từ QLabel
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        img = pixmap.toImage()
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        current_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

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
                if shape["type"] in ["ellipse", "donut"]:
                    # Tính bounding box mới sau khi xoay
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
                    # Tạo mask cho ellipse đã xoay
                    mask = np.zeros_like(current_img)
                    if shape["type"] == "ellipse":
                        cv2.ellipse(mask, center, axes, shape["angle"], 0, 360, (255, 255, 255), -1)
                    elif shape["type"] == "donut":
                        outer_axes = axes
                        inner_axes = (
                            max(1, outer_axes[0] - shape["thickness"]),
                            max(1, outer_axes[1] - shape["thickness"])
                        )
                        cv2.ellipse(mask, center, outer_axes, shape["angle"], 0, 360, (255, 255, 255), -1)
                        cv2.ellipse(mask, center, inner_axes, shape["angle"], 0, 360, (0, 0, 0), -1)
                    # Áp dụng mask lên ảnh gốc
                    result = cv2.bitwise_and(current_img, mask)
                    result[mask == 0] = 0
                    # Lưu ảnh
                    cv2.imwrite(f"shape_{i}.png", result)
                else:
                    cropped_img = current_img[y1:y2, x1:x2].copy()
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
                    cv2.imwrite(f"shape_{i}.png", result)

    def closeEvent(self, event):
        # Giải phóng camera khi đóng ứng dụng
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())