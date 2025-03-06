import cv2
from img_pro1 import ImageProcessor

def main():
    processor = ImageProcessor()
    cropped_images = []

    while True:

        new_cropped_images = processor.process_frame()
        if new_cropped_images is not None:
            cropped_images = new_cropped_images

        # Hiển thị các ảnh đã cắt liên tục
        for i, img in enumerate(cropped_images):
            cv2.imshow(f"Cropped Image {i}", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            processor.toggle_camera_display()
        elif key == ord('q'):
            break

    processor.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()