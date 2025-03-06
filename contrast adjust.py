from PIL import Image, ImageEnhance

img = Image.open("contrast.jpg")  # Mở ảnh
enhancer = ImageEnhance.Brightness(img)
img_bright = enhancer.enhance(1)  # Tăng độ sáng lên 1.5 lần
img_bright.show()

enhancer = ImageEnhance.Contrast(img)
img_contrast = enhancer.enhance(1)  # Tăng độ tương phản
img_contrast.show()

enhancer = ImageEnhance.Color(img)
img_color = enhancer.enhance(1)  # Tăng cường độ màu lên 2 lần
img_color.show()

enhancer = ImageEnhance.Sharpness(img)
img_sharp = enhancer.enhance(1)  # Tăng độ sắc nét lên 2 lần
img_sharp.show()

from PIL import ImageFilter
img_edges = img.filter(ImageFilter.EDGE_ENHANCE)  # Làm nét biên
img_edges.show()
import numpy as np
from PIL import Image

# Mở ảnh
img = Image.open("contrast1.jpg").convert("RGB")
img_array = np.array(img)

# === Người dùng chọn vùng ảnh ===
x1, y1, x2, y2 = 50, 50, 100, 100  # Thay đổi tọa độ theo nhu cầu

# Trích xuất vùng ảnh đã chọn
selected_region = img_array[y1:y2, x1:x2]  # Chú ý chỉ số dòng trước, cột sau

# Tính giá trị RGB trung bình của vùng đã chọn
mean_color = np.mean(selected_region, axis=(0, 1)).astype(int)
print("Giá trị RGB đặc trưng:", mean_color)  # Debugging

# Ngưỡng sai lệch (có thể điều chỉnh)
threshold = 20

# Tạo mặt nạ (mask) xác định pixel nào cần đổi màu
mask = np.all(np.abs(img_array - mean_color) <= threshold, axis=-1)

# Tạo ảnh mới với nền đen
output_array = np.zeros_like(img_array)

# Đổi màu pixel thỏa điều kiện sang màu đỏ
output_array[mask] = [255, 0, 0]

# Chuyển NumPy array về ảnh và hiển thị/lưu kết quả
new_img = Image.fromarray(output_array)
new_img.show()
new_img.save("output.jpg")




