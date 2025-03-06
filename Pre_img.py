from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class ImageProcessor:
    def __init__(self, input_img):
        self.image = Image.open(input_img)
        self.image = self.image.convert("RGB")  # Đảm bảo ảnh ở chế độ RGB

    def apply_gaussian_blur(self, radius=1):
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))

    def enhance_brightness(self, factor=1.5):
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)

    def enhance_contrast(self, factor=2):
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)

    def enhance_sharpness(self, factor=3):
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(factor)

    def enhance_edges(self):
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE)

    def apply_contour(self):
        self.image = self.image.filter(ImageFilter.CONTOUR)

    def convert_to_grayscale(self):
        self.image = self.image.convert("L")

    def apply_threshold(self, threshold=50):
        self.image = self.image.point(lambda p: 255 if p > threshold else 0)

    def enhance_color(self, factor=3):
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(factor)

    def select_region(self, x1, y1, x2, y2):
        img_array = np.array(self.image)
        selected_region = img_array[y1:y2, x1:x2]
        return selected_region

    def calculate_mean_color(self, region):
        return np.mean(region, axis=(0, 1)).astype(int)

    def apply_color_mask(self, mean_color, threshold=20):
        img_array = np.array(self.image)
        mask = np.all(np.abs(img_array - mean_color) <= threshold, axis=-1)
        output_array = np.zeros_like(img_array)
        output_array[mask] = [255, 0, 0]
        self.image = Image.fromarray(output_array)

    def show_image(self):
        self.image.show()

    def save_image(self, output_path):
        self.image.save(output_path)


# Sử dụng class ImageProcessor
if __name__ == "__main__":
    processor = ImageProcessor("image.jpg")

    # Áp dụng các bước xử lý ảnh
    processor.apply_gaussian_blur(1)
    processor.enhance_brightness(1.5)
    processor.enhance_contrast(2)
    processor.enhance_sharpness(3)
    processor.enhance_edges()
    processor.apply_contour()
    processor.convert_to_grayscale()
    processor.apply_threshold(50)
    processor.enhance_color(3)

    # Xử lý vùng ảnh và đổi màu
    selected_region = processor.select_region(50, 50, 100, 100)
    mean_color = processor.calculate_mean_color(selected_region)
    processor.apply_color_mask(mean_color, threshold=20)

    # Hiển thị và lưu ảnh
    processor.show_image()
    processor.save_image("output.jpg")