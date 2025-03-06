import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, util, morphology, segmentation, restoration, feature, transform
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.filters import gaussian, sobel, laplace, unsharp_mask
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, inpaint
from skimage.feature import canny
from skimage.transform import rotate, resize, warp, AffineTransform
from skimage.util import random_noise
from torchvision import transforms
import torch
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
# from tf.keras.applications.resnet50 import preprocess_input as resnet_preprocess
# from tf.keras.applications.vgg16 import preprocess_input as vgg_preprocess
# from tf.keras.applications.inception_v3 import preprocess_input as inception_preprocess

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
        result[:, :, i] = exposure.equalize_hist(image[:, :, i])
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
    return denoise_nl_means(image, h=0.1, fast_mode=True)

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

# 9. Pre-trained Model Preprocessing (Chuẩn hóa cho mô hình ResNet, VGG, Inception)
# def preprocess_for_pretrained_model(image, model_name='resnet'):
#     if model_name == 'resnet':
#         return resnet_preprocess(image.copy())
#     elif model_name == 'vgg':
#         return vgg_preprocess(image.copy())
#     elif model_name == 'inception':
#         return inception_preprocess(image.copy())
#     return image

# 9. Super-resolution (Siêu phân giải)
def super_resolution(image):
    # Sử dụng mô hình EDSR (cần cài đặt OpenCV contrib)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('EDSR_x4.pb')  # Tải mô hình EDSR
    sr.setModel('edsr', 4)  # Chọn hệ số scale
    return sr.upsample(image)

# 10. Thay đổi kích thước ảnh
def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

# 3. Padding (thêm viền)
def add_padding(image, top, bottom, left, right, color=(0, 0, 0)):
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# 4. Cắt ảnh (Cropping)
def crop_image(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

# 5. Chuyển đổi không gian màu
def convert_color_space(image, mode='gray'):
    if mode == 'gray':
        return rgb2gray(image)
    elif mode == 'hsv':
        return rgb2hsv(image)
    elif mode == 'lab':
        return rgb2lab(image)
    return image

# 6. Làm mờ ảnh
def blur_image(image, method='gaussian', kernel_size=5):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    return image

# 7. Khử nhiễu
def denoise_image(image, method='nlm'):
    if method == 'nlm':
        return denoise_nl_means(image, h=0.1, fast_mode=True)
    elif method == 'tv':
        return denoise_tv_chambolle(image, weight=0.1)
    return image

# 8. Phát hiện biên
def detect_edges(image, method='canny'):
    if method == 'canny':
        return canny(image, sigma=1)
    elif method == 'sobel':
        return sobel(image)
    elif method == 'laplace':
        return laplace(image)
    return image

# 19. Phép toán hình thái học
def apply_morphology(image, operation='erosion', kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erosion':
        return erosion(image, kernel)
    elif operation == 'dilation':
        return dilation(image, kernel)
    elif operation == 'opening':
        return opening(image, kernel)
    elif operation == 'closing':
        return closing(image, kernel)
    return image

# 20. Xoay ảnh
def rotate_image(image, angle):
    return rotate(image, angle, resize=True)

# 21. Phân ngưỡng ảnh
def threshold_image(image, method='binary', thresh=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == 'binary':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

# 12. Tăng cường ảnh (Histogram Equalization)
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)

# 23. Thêm nhiễu
def add_noise(image, mode='gaussian'):
    if mode == 'gaussian':
        return util.random_noise(image, mode='gaussian')
    elif mode == 's&p':
        return util.random_noise(image, mode='s&p')
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

# 16. Làm sắc nét ảnh (Unsharp Masking)
def sharpen_image(image):
    return unsharp_mask(image, radius=5, amount=2)

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
# thêm xử lý
#31 xóa phông ảnh
def remove_background(frame):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    mask = bg_subtractor.apply(frame)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result
#32 lọc sobel
def sobel_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.convertScaleAbs(sobelx + sobely)

def split_channels(img):
    b, g, r = cv2.split(img)
    return b, g, r

def detect_shapes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
    return img

def edge_detection(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

def gradient_edge_detection(img):
    kernel = np.ones((5,5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return cv2.drawKeypoints(img, keypoints, None)

def extract_orb_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return cv2.drawKeypoints(img, keypoints, None)

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
# preprocessed_resnet = preprocess_for_pretrained_model(image, 'resnet')
# preprocessed_vgg = preprocess_for_pretrained_model(image, 'vgg')
# preprocessed_inception = preprocess_for_pretrained_model(image, 'inception')
super_res = super_resolution(image)

# teem xu lý
# Apply processing functions
bg_removed = remove_background(image)
sobel_img = sobel_filter(image)
b, g, r = split_channels(image)
shapes_img = detect_shapes(image.copy())
edges_img = edge_detection(image)
gradient_img = gradient_edge_detection(image)
features_img = extract_features(image.copy())
orb_features_img = extract_orb_features(image.copy())
hog_features = extract_hog_features(image)

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
# show_image(preprocessed_resnet, 'Preprocessed for ResNet')
# show_image(preprocessed_vgg, 'Preprocessed for VGG')
# show_image(preprocessed_inception, 'Preprocessed for Inception')
show_image(super_res, 'Super-Resolution (EDSR x4)')

# Display results
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

