import cv2
from image_processor import ImageProcessor
import time

def main():
    # Khởi tạo đối tượng ImageProcessor
    processor = ImageProcessor()

    # Bắt đầu xử lý ảnh từ camera và nhận các ảnh đã cắt
    cropped_images = processor.process()

    # Hiển thị các ảnh đã cắt

    for i, img in enumerate(cropped_images):
        cv2.imshow(f"Cropped Image {i}", img)
        print(i)
        time.sleep(5)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()