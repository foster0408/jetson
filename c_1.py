from PIL import Image, ImageEnhance
from PIL import ImageFilter

img = Image.open("image.jpg")  # Mở ảnh
img = img.filter(ImageFilter.GaussianBlur(1)) 
enhancer = ImageEnhance.Brightness(img)
img_bright = enhancer.enhance(1)  # Tăng độ sáng lên 1.5 lần
img_bright.show()

enhancer = ImageEnhance.Contrast(img)
img_contrast = enhancer.enhance(2)  # Tăng độ tương phản
img_contrast.show()



enhancer = ImageEnhance.Sharpness(img)
img_sharp = enhancer.enhance(3)  # Tăng độ sắc nét lên 2 lần
img_sharp.show()


img_edges = img_contrast.filter(ImageFilter.EDGE_ENHANCE)  # Làm nét biên
img_edges.show()
CONTOUR = img_edges.filter(ImageFilter.CONTOUR)  # Làm nét biên
CONTOUR.show()

img_gray= CONTOUR.convert("L")
thres = 50
img_out= img_gray.point(lambda p: 255 if p > thres else 0)
img_out.show()

enhancer = ImageEnhance.Color(CONTOUR)
img_color = enhancer.enhance(3)  # Tăng cường độ màu lên 2 lần
img_color.show()
import numpy as np
from PIL import Image

# Mở ảnh
img = Image.open("image.jpg").convert("RGB")
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




