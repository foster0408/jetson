import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào (chuyển sang ảnh xám)
img = cv2.imread('contrast.jpg', 0)
reference_img = cv2.imread('contrast1.jpg', 0)

# 1. Cân bằng biểu đồ (HE)
def histogram_equalization(img):
    return cv2.equalizeHist(img)

# 2. Adaptive Histogram Equalization (AHE)
def adaptive_histogram_equalization(img, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

# 3. Hiệu chỉnh Gamma
def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# 4. Bộ lọc thông cao
def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  16, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

# 5. Giãn tương phản (Contrast Stretching)
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return cv2.normalize(img, None, alpha=p2, beta=p98, norm_type=cv2.NORM_MINMAX)

# 6. Logarithmic/Exponential Transform
def logarithmic_transform(img):
    img = img.astype(np.float32) + 1
    c = 255 / np.log(1 + np.max(img))
    log_image = c * np.log(img)
    return log_image.astype(np.uint8)

def exponential_transform(img):
    img = img.astype(np.float32)
    c = 255 / (1 + np.exp(-1))
    exp_image = c * (1 / (1 + np.exp(-(img/255 - 0.5)*10)))
    return exp_image.astype(np.uint8)

# 7. S-Curve Adjustment
def s_curve_adjustment(img):
    x = np.linspace(0, 1, 256)
    s_curve = 0.5 * np.sin(1.5 * np.pi * (x - 0.5)) + 0.5
    s_curve = (s_curve - s_curve.min()) / (s_curve.max() - s_curve.min())
    return cv2.LUT(img, (s_curve * 255).astype(np.uint8))

# 8. Unsharp Masking
def unsharp_masking(img, radius=5, amount=2):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

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

    # DFT using OpenCV
    dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Tạo bộ lọc
    H = np.zeros((rows, cols, 2), np.float32)
    center_x, center_y = rows//2, cols//2
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            H[u, v] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2))) + gamma_l

    # Áp dụng bộ lọc
    filtered_dft = dft_shift * H
    idft_shift = np.fft.ifftshift(filtered_dft)
    img_filtered = cv2.idft(idft_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Exponential và chuẩn hóa
    result = np.exp(img_filtered) - 1
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 11. Multi-Scale Enhancement
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

# 12. Điều chỉnh không gian màu HSV
def color_space_adjustment(img):
    img_color = cv2.cvtColor(cv2.imread('contrast.jpg'), cv2.COLOR_BGR2HSV)
    img_color[:, :, 2] = cv2.equalizeHist(img_color[:, :, 2])
    return cv2.cvtColor(img_color, cv2.COLOR_HSV2RGB)

# 13. Phương pháp hình thái học
def morphological_contrast(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return cv2.add(cv2.subtract(img, tophat), blackhat)

# 14. Histogram Specification
def histogram_matching(img, reference_img):
    # Tính histogram của ảnh gốc và ảnh tham chiếu
    src_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    ref_hist = cv2.calcHist([reference_img], [0], None, [256], [0, 256])

    # Tính CDF
    src_cdf = np.cumsum(src_hist) / img.size
    ref_cdf = np.cumsum(ref_hist) / reference_img.size

    # Tạo LUT
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.argmin(np.abs(ref_cdf - src_cdf[i]))

    return cv2.LUT(img, lut)

# Hiển thị kết quả
methods = {
    'Original': img,
    'CLAHE': adaptive_histogram_equalization(img),
    'Retinex': retinex(img),
    'homomorphic_filter': homomorphic_filter(img),
    'color_space_adjustment': color_space_adjustment(img),
    'morphological_contrast': morphological_contrast(img),
    'histogram_matching': histogram_matching(img, reference_img),
    'high_pass_filter': high_pass_filter(img),
}

plt.figure(figsize=(20, 10))
for i, (title, result) in enumerate(methods.items()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(result, cmap='gray' if len(result.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()