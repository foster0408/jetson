import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
# import torch
# import tensorflow as tf
#
# print(tf.__version__)
# print(tf.keras.__version__)

# Đọc ảnh
image = cv2.imread('input.jpg')  # Thay 'input.jpg' bằng đường dẫn ảnh của bạn
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

# 1. Chuẩn hóa cường độ ảnh
def normalize_image(image):
    return image / 255.0

# 2. Standardization (Z-score Normalization)
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

# 3. Clipping (Giới hạn giá trị pixel)
def clip_image(image, min_val=0, max_val=255):
    return np.clip(image, min_val, max_val)

# 4. Cân bằng màu (Color Balance)
def color_balance(image):
    result = np.zeros_like(image)
    for i in range(3):  # Áp dụng cho từng kênh màu (R, G, B)
        result[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return result

# 5. Giảm số lượng màu (Color Quantization)
def color_quantization(image, n_colors=64):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(image.shape)

# 6. Non-local Means Denoising
def non_local_means_denoising(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 7. Scaling (Co giãn ảnh)
def scale_image(image, scale_factor=0.5):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# 8. GrabCut (Phân vùng ảnh)
def grabcut_segmentation(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)  # ROI
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask2[:, :, np.newaxis]

# 9. Super-resolution (Siêu phân giải)
def super_resolution(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('EDSR_x4.pb')  # Tải mô hình EDSR
    sr.setModel('edsr', 4)  # Chọn hệ số scale
    return sr.upsample(image)
#10 Phát hiện góc corner
def detect_harris_corners(img, block_size=2, ksize=3, k=0.04):
    """Phát hiện góc bằng Harris Corner"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    img_copy = img.copy()
    img_copy[dst > 0.01 * dst.max()] = [0, 0, 255]  # Đánh dấu góc màu đỏ
    return img_copy
#11 phát hiện góc shitomasi
def detect_shitomasi_corners(img, max_corners=25, quality_level=0.01, min_distance=10):
    """Phát hiện góc bằng Shi-Tomasi"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

    if corners is not None:  # Kiểm tra nếu có góc nào được phát hiện
        corners = np.int64(corners)
        img_copy = img.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)  # Đánh dấu góc màu xanh
        return img_copy
    else:
        print("Không tìm thấy góc nào.")
        return img  # Trả về ảnh gốc nếu không có góc
#11 Phát hiện đối tượng ORB
def detect_orb_keypoints(img):
    """Trích xuất đặc trưng bằng ORB"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    img_copy = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    return img_copy
# 12. Thay đổi kích thước ảnh
def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

# 13. Padding (thêm viền)
def add_padding(image, top, bottom, left, right, color=(0, 0, 0)):
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# 14. Cắt ảnh (Cropping)
def crop_image(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

# 15. Chuyển đổi không gian màu
def convert_color_space(image, mode='gray'):
    if mode == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif mode == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif mode == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return image

# 16. Làm mờ ảnh
def blur_image(image, method='gaussian', kernel_size=5):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    return image

# 17. Khử nhiễu
def denoise_image(image, method='nlm'):
    if method == 'nlm':
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif method == 'tv':
        return cv2.ximgproc.createFastGlobalSmootherFilter(image, lambda_=0.1).filter(image)
    return image

# 18. Phát hiện biên
def detect_edges(image, method='canny'):
    if method == 'canny':
        return cv2.Canny(image, 100, 200)
    elif method == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.convertScaleAbs(sobelx + sobely)
    elif method == 'scharr':
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(scharrx + scharry)
    elif method == 'laplace':
        return cv2.Laplacian(image, cv2.CV_64F)
    elif method == 'prewitt':
        prewittx = cv2.filter2D(image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitty = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        return cv2.convertScaleAbs(prewittx + prewitty)
    elif method == 'roberts':
        roberts_cross_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_cross_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        robertsx = cv2.filter2D(image, -1, roberts_cross_x)
        robertsy = cv2.filter2D(image, -1, roberts_cross_y)
        return cv2.convertScaleAbs(robertsx + robertsy)
    else:
        raise ValueError("Phương pháp phát hiện cạnh không hợp lệ.")


# 19. Phép toán hình thái học
def apply_morphology(image, operation='erosion', kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == 'erosion':
        return cv2.erode(image, kernel)
    elif operation == 'dilation':
        return cv2.dilate(image, kernel)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    elif operation == 'tophat':
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif operation == 'blackhat':
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError("Phép toán hình thái học không hợp lệ.")


# 20. Xoay ảnh
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# 21. Phân ngưỡng ảnh
def threshold_image(image, method='binary', thresh=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == 'binary':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif method == 'binary_inv':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    elif method == 'trunc':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_TRUNC)
    elif method == 'tozero':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_TOZERO)
    elif method == 'tozero_inv':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_TOZERO_INV)
    elif method == 'otsu':
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_mean':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == 'adaptive_gaussian':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        raise ValueError("Phương pháp threshold không hợp lệ.")

    return result

# 22. Tăng cường ảnh (Histogram Equalization)
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)

# 23. Thêm nhiễu
def add_noise(image, mode='gaussian'):
    if mode == 'gaussian':
        noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    elif mode == 's&p':
        noise = np.random.randint(0, 2, image.shape, dtype=np.uint8) * 255
        return cv2.add(image, noise)
    return image

# 24. Khôi phục ảnh (Inpainting)
def inpaint_image(image, mask):
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 25. Data Augmentation (sử dụng torchvision)
def augment_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor()
    ])
    return transform(image)

# 26. Làm sắc nét ảnh (Unsharp Masking)
def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# 27. Biến đổi hình học (Affine Transform)
def affine_transform(image):
    rows, cols, _ = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (cols, rows))

# 28. Biến đổi phối cảnh (Perspective Transform)
def perspective_transform(image):
    rows, cols, _ = image.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (300, 300))

# 29. Phân vùng ảnh (Watershed)
def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image

# 30. Tăng cường độ tương phản (CLAHE)
def clahe_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
#31 xoa nền ảnh
def remove_background(frame):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    mask = bg_subtractor.apply(frame)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result
#32 loc sobel
def sobel_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.convertScaleAbs(sobelx + sobely)
#33 tách ảnh
def split_channels(img):
    b, g, r = cv2.split(img)
    return b, g, r
#34 phát hiện hình học
def detect_shapes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
    return img
#35 phát hiện cạnh
def edge_detection(img):
    edges = cv2.Canny(img, 100, 200)
    return edges
#36 Phát hiện cạnh grandient
def gradient_edge_detection(img):
    kernel = np.ones((5,5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient
#37 Phát hiện đối tiowngj SIFT
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return cv2.drawKeypoints(img, keypoints, None)
#37 phát hiện ORB
def extract_orb_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return cv2.drawKeypoints(img, keypoints, None)
#38 phát hiện HOG
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    return hog_features

# Hiển thị ảnh
def show_image(image, title='Image'):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Áp dụng các kỹ thuật
normalized = normalize_image(image)
resized = resize_image(image, 300, 300)
padded = add_padding(image, 50, 50, 50, 50)
cropped = crop_image(image, 100, 100, 400, 400)
gray = convert_color_space(image, 'gray')
blurred = blur_image(image, 'gaussian')
denoised = denoise_image(image, 'nlm')
edges = detect_edges(gray, 'canny')
morphed = apply_morphology(gray, 'erosion')
rotated = rotate_image(image, 45)
thresholded = threshold_image(image, 'otsu')
enhanced = enhance_image(image)
noisy = add_noise(image, 'gaussian')
augmented = augment_image(image)
sharpened = sharpen_image(image)
affine = affine_transform(image)
perspective = perspective_transform(image)
watershed = watershed_segmentation(image)
clahe = clahe_enhancement(image)

normalized = normalize_image(image)
standardized = standardize_image(image)
clipped = clip_image(image, 50, 200)
balanced = color_balance(image)
quantized = color_quantization(image, 32)
denoised_nlm = non_local_means_denoising(image)
scaled = scale_image(image, 0.75)
grabcut = grabcut_segmentation(image)
super_res = super_resolution(image)

# Bổ sung các thuật toán
bg_removed = remove_background(image)
sobel_img = sobel_filter(image)
b, g, r = split_channels(image)
shapes_img = detect_shapes(image.copy())
edges_img = edge_detection(image)
gradient_img = gradient_edge_detection(image)
features_img = extract_features(image.copy())
orb_features_img = extract_orb_features(image.copy())
#hog_features = extract_hog_features(image)

harris_corners = detect_harris_corners(image, block_size=2, ksize=3, k=0.04)
shitomasi_corners = detect_shitomasi_corners(image, max_corners=25, quality_level=0.01, min_distance=10)
detect_orb = detect_orb_keypoints(image)

# Hiển thị kết quả
show_image(image, 'Original')
show_image(normalized, 'Normalized')
show_image(resized, 'Resized')
show_image(padded, 'Padded')
show_image(cropped, 'Cropped')
show_image(gray, 'Grayscale')
show_image(blurred, 'Blurred')
show_image(denoised, 'Denoised')
show_image(edges, 'Edges')
show_image(morphed, 'Morphed')
show_image(rotated, 'Rotated')
show_image(thresholded, 'Thresholded')
show_image(enhanced, 'Enhanced')
show_image(noisy, 'Noisy')
show_image(augmented.permute(1, 2, 0).numpy(), 'Augmented')
show_image(sharpened, 'Sharpened')
show_image(affine, 'Affine Transform')
show_image(perspective, 'Perspective Transform')
show_image(watershed, 'Watershed Segmentation')
show_image(clahe, 'CLAHE Enhancement')

# Hiển thị kết quả
show_image(image, 'Original')
show_image(normalized, 'Normalized')
show_image(standardized, 'Standardized')
show_image(clipped, 'Clipped')
show_image(balanced, 'Color Balanced')
show_image(quantized, 'Color Quantization (32 colors)')
show_image(denoised_nlm, 'Non-local Means Denoising')
show_image(scaled, 'Scaled (0.75x)')
show_image(grabcut, 'GrabCut Segmentation')
show_image(super_res, 'Super-Resolution (EDSR x4)')

show_image(bg_removed, 'Background Removed')
show_image(sobel_img, 'Sobel Filter')
show_image(r, 'Red Channel')
show_image(g, 'Green Channel')
show_image(b, 'Blue Channel')
show_image(shapes_img, 'Shape Detection')
show_image(edges_img, 'Edge Detection')
show_image(gradient_img, 'Gradient Edge Detection')
show_image(features_img, 'Feature Extraction (SIFT)')
show_image(orb_features_img, 'Feature Extraction (ORB)')
#show_image(hog_features, 'hog_features(HOG)')

show_image(harris_corners, 'harris_corners ()')
show_image(shitomasi_corners, 'shitomasi_corners ()')
show_image(detect_orb, 'detect_orb (ORB)')
