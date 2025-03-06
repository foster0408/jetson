import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame
from PyQt5.QtGui import QPixmap  # Cho PyQt5
# Hoặc
#from PySide2.QtGui import QPixmap  # Cho PySide2

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QWheelEvent

from PyQt5.QtCore import Qt
from window_1 import (Ui_MainWindow)  # Import file UI đã chuyển đổi

class CustomFrame(QFrame):
    def mousePressEvent(self, event):
        # Xử lý khi nhấn chuột
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
           # x = 640
            #y = 480
            print(f"Tọa độ chuột: ({x}, {y})")
            # Hoặc hiển thị lên widget khác (ví dụ: self.label)
        super().mousePressEvent(event)  # Gọi phương thức gốc


class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = QPixmap()  # Lưu ảnh gốc
        self.scale_factor = 1.0  # Tỉ lệ zoom hiện tại
        self.min_scale = 0.1  # Zoom tối thiểu
        self.max_scale = 10.0  # Zoom tối đa
        self.setAlignment(Qt.AlignCenter)  # Căn giữa ảnh

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        super().setPixmap(pixmap.scaled(pixmap.size() * self.scale_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def wheelEvent(self, event: QWheelEvent):
        # Xác định hướng lăn chuột
        delta = event.angleDelta().y()
        if delta > 0:
            new_scale = self.scale_factor * 1.1  # Zoom in
        else:
            new_scale = self.scale_factor / 1.1  # Zoom out

        # Giới hạn tỉ lệ zoom
        new_scale = max(self.min_scale, min(new_scale, self.max_scale))
        if new_scale == self.scale_factor:
            return  # Không thay đổi

        # Lấy vị trí chuột tương đối so với label
        mouse_pos = event.position()
        old_pos_rel_label = (mouse_pos.x(), mouse_pos.y())

        # Cập nhật tỉ lệ zoom
        self.scale_factor = new_scale

        # Scale ảnh và cập nhật label
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)
        self.adjustSize()  # Điều chỉnh kích thước label

        # Điều chỉnh vị trí cuộn để giữ ảnh tập trung vào chuột
        scroll_area = self.parent().parent()  # Truy cập QScrollArea
        if isinstance(scroll_area, QScrollArea):
            # Tính toán vị trí cuộn mới
            new_hscroll = (old_pos_rel_label[0] * self.scale_factor) - (scroll_area.viewport().width() / 2)
            new_vscroll = (old_pos_rel_label[1] * self.scale_factor) - (scroll_area.viewport().height() / 2)

            scroll_area.horizontalScrollBar().setValue(int(new_hscroll))
            scroll_area.verticalScrollBar().setValue(int(new_vscroll))


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Kết nối sự kiện chuột với các phương thức xử lý
        self.camera02frame.mousePressEvent = self.label_mouse_press_event
        self.camera02frame.mouseReleaseEvent = self.label_mouse_release_event
        self.camera02frame.mouseMoveEvent = self.label_mouse_move_event
        #self.camera02frame = CustomFrame(self)
        #self.camera02frame.addWidget(self.camera02frame)
        self.camera02frame.setFixedSize(640, 480)  # Kích thước cố định: 200x100 px

        # Hoặc:
        #self.camera02frame.resize(300, 150)  # Kích thước động
        # Hoặc:
        self.camera02frame.setGeometry(300 , 100 , 1640 , 480 )  # Vị trí (50,50), kích thước 250x80

        # Tự động điều chỉnh theo nội dung
        #self.camera02frame.setText("Nội dung mới dài hơn")
        self.camera02frame.adjustSize()
        self.scroll_area = QScrollArea()
        self.camera02frame = ZoomableLabel()
        self.camera02frame.setPixmap(QPixmap("/home/vqbg/picture2_5.jpg"))
        #self.camera02frame.setScaledContents(True)
        self.scroll_area.setWidget(self.camera02frame)
        self.setCentralWidget(self.scroll_area)

    def label_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.camera02frame.setText("Left button pressed")
        elif event.button() == Qt.RightButton:
            self.camera02frame.setText("Right button pressed")

    def label_mouse_release_event(self, event):
        self.camera02frame.setText("Mouse button released")

    def label_mouse_move_event(self, event):
        self.camera02frame.setText(f"Mouse moved to ({event.x()}, {event.y()})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())