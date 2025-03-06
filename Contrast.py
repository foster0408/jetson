import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import unsharp_mask
from scipy.fftpack import fft2, ifft2, fftshift
import pywt

# Đọc ảnh đầu vào (chuyển sang ảnh xám)
img = cv2.imread('contrast.jpg', 0)  # Thay 'input.jpg' bằng đường dẫn ảnh của bạn
reference_img = cv2.imread('contrast1.jpg', 0)

# 1. Cân bằng biểu đồ (HE)
def histogram_equalization(img):
    return cv2.convertScaleAbs(img, alpha=1, beta=150)

#def histogram_equalization(img):
    #return cv2.equalizeHist(img)

# 2. Adaptive Histogram Equalization (AHE)
def adaptive_histogram_equalization(img, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


# 3. CLAHE (giới hạn tương phản)
# Đã implement ở phương pháp 2 với tham số clip_limit

# 4. Hiệu chỉnh Gamma
def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                        [-1,  16, -1],
                        [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)
# 7. Wavelet Transform-based Enhancemen0
def wavelet_transform(img):
    coeffs = pywt.wavedec2(img, 'haar', level=2)
    coeffs[0] *= 0.5  # Enhance approximation coefficients
    return pywt.waverec2(coeffs, 'haar').astype(np.uint8)

# 8. Fourier Transform-based Enhancement
def fourier_transform(img):
    f = fft2(img)
    fshift = fftshift(f)
    magnitude_spectrum = 1 * np.log(np.abs(fshift))
    return magnitude_spectrum.astype(np.uint8)

# 9. Deep Learning-based Enhancement
# def deep_learning_enhancement(image, model_path="contrast_model.h5"):
#     model = load_model(model_path)
#     image = cv2.resize(image, (128, 128)) / 255.0  # Resize and normalize
#     image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimensions
#     enhanced_image = model.predict(image)[0, :, :, 0] * 255  # Predict and rescale
#     return enhanced_image.astype(np.uint8)
# 5. Giãn tương phản (Contrast Stretching)
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


# 6. Logarithmic/Exponential Transform
def logarithmic_transform(img):
    return np.array(255 * (np.log(img + 1) / np.log(np.max(img + 1))), dtype=np.uint8)


def exponential_transform(img):
    return np.array(255 * (img ** 0.5 / np.max(img ** 0.5)), dtype=np.uint8)


# 7. S-Curve Adjustment
def s_curve_adjustment(img):
    x = np.linspace(0, 1, 256)
    s_curve = 0.5 * np.sin(1.5 * np.pi * (x - 0.5)) + 0.5
    s_curve = (s_curve - s_curve.min()) / (s_curve.max() - s_curve.min())
    return cv2.LUT(img, (s_curve * 255).astype(np.uint8))


# 8. Unsharp Masking
def unsharp_masking(img, radius=5, amount=2):
    return unsharp_mask(img, radius=radius, amount=amount, channel_axis=False)


# 9. Retinex (Simplified McCann99)
def retinex(img, sigma=2):
    img = img.astype(np.float32) + 1.0
    log_img = np.log(img)
    log_illumination = cv2.GaussianBlur(log_img, (0, 0), sigma)
    retinex_output = log_img - log_illumination
    return cv2.normalize(retinex_output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 10. Homomorphic Filtering
def homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, c=1):
    rows, cols = img.shape
    img_log = np.log(img.astype(np.float32) + 1e-5)

    # DFT
    dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Tạo bộ lọc
    H = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - rows / 2) ** 2 + (v - cols / 2) ** 2)
            H[u, v] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2))) + gamma_l

    # Áp dụng bộ lọc
    filtered_dft = dft_shift * H
    idft_shift = np.fft.ifftshift(filtered_dft)
    img_filtered = cv2.idft(idft_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Exponential và chuẩn hóa
    result = np.exp(img_filtered) - 1
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 11. Multi-Scale Enhancement (Ví dụ với Laplacian Pyramid)
def multi_scale_enhancement(img):
    # Tạo Gaussian pyramid
    G = img.copy()
    gp = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gp.append(G)

    # Tạo Laplacian pyramid
    lp = [gp[-1]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)

    # Tăng cường từng mức
    lp_enhanced = [cv2.addWeighted(l, 1.5, l, 0, 0) for l in lp]

    # Tái tạo ảnh
    img_reconstructed = lp_enhanced[-1]
    for i in range(4, -1, -1):
        img_reconstructed = cv2.pyrUp(img_reconstructed)
        img_reconstructed = cv2.add(img_reconstructed, lp_enhanced[i])

    return cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 12. Machine Learning (Ví dụ với Autoencoder đơn giản)
# Yêu cầu thư viện Keras/TensorFlow
"""
from keras.layers import Input, Conv2D
from keras.models import Model

def build_simple_autoencoder():
    inputs = Input(shape=(None, None, 1))
    x = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    return model

# Huấn luyện mô hình trên tập dữ liệu ảnh của bạn trước khi sử dụng
"""


# 13. Điều chỉnh không gian màu (Ví dụ với HSV)
def color_space_adjustment(img):
    img_color = cv2.cvtColor(cv2.imread('contrast.jpg'), cv2.COLOR_BGR2HSV)
    img_color[:, :, 2] = cv2.equalizeHist(img_color[:, :, 2])
    return cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)


# 14. Phương pháp hình thái học (Top-Hat/Bottom-Hat)
def morphological_contrast(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return cv2.add(cv2.subtract(img, tophat), blackhat)


# 15. Histogram Specification
def histogram_matching(img, reference_img):
    return exposure.match_histograms(img, reference_img)


# Hiển thị kết quả
methods = {
    'Original': img,
    'HE': histogram_equalization(img),
    #'CLAHE': adaptive_histogram_equalization(img),
    #'Gamma 0.5': gamma_correction(img, 0.5),
    #'Contrast Stretching': contrast_stretching(img),
    #'S-Curve': s_curve_adjustment(img),
    #'Unsharp Masking': unsharp_masking(img),
    #'Retinex': retinex(img),
    #'homomorphic_filter': homomorphic_filter(img),
    #'multi_scale_enhancement': multi_scale_enhancement(img),
    'color_space_adjustment': color_space_adjustment(img),
    'morphological_contrast': morphological_contrast(img),
    'histogram_matching': histogram_matching(img, reference_img),
    'high_pass_filter(img)': high_pass_filter(img),
    #'wavelet_transform(img)': wavelet_transform(img),
    #'fourier_transform(img)': fourier_transform(img),
}

plt.figure(figsize=(20, 10))
for i, (title, result) in enumerate(methods.items()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()