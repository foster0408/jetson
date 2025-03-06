import numpy as np
import cv2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import NMF
import pywt
import bm3d
from sklearn.decomposition import DictionaryLearning
from sklearn.cluster import KMeans
import pywt.data
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from scipy import signal
from scipy.signal import convolve2d
# 1.Wiener Filter là một phương pháp lọc thích ứng,
# giúp giảm nhiễu trong ảnh bằng cách ước lượng giá trị của các pixel với một trung bình có trọng số.
def wiener_filter(image, size=3):
    kernel = np.ones((size, size)) / (size * size)
    filtered = convolve2d(image, kernel, boundary='symm', mode='same')
    noise = image - filtered
    return filtered + noise.mean()  # Trả lại ảnh với nhiễu giảm bớt
#2. Anisotropic Diffusion là một kỹ thuật làm mượt ảnh trong khi giữ lại các cạnh sắc nét.
# Thường được gọi là "Canny Diffusion" hoặc "Perona-Malik Filter".
def anisotropic_diffusion(image, niter=15, kappa=50, gamma=0.1):
    img = np.float32(image)
    for _ in range(niter):
        grad_north = np.roll(img, 1, axis=0) - img
        grad_south = np.roll(img, -1, axis=0) - img
        grad_east = np.roll(img, 1, axis=1) - img
        grad_west = np.roll(img, -1, axis=1) - img
        conduction_north = np.exp(-(grad_north / kappa) ** 2)
        conduction_south = np.exp(-(grad_south / kappa) ** 2)
        conduction_east = np.exp(-(grad_east / kappa) ** 2)
        conduction_west = np.exp(-(grad_west / kappa) ** 2)
        img += gamma * (conduction_north * grad_north + conduction_south * grad_south +
                        conduction_east * grad_east + conduction_west * grad_west)
    return np.uint8(img)
#3. Bilateral Filter là một phương pháp lọc mạnh mẽ giúp giữ các cạnh sắc nét trong khi làm mờ
# các chi tiết không cần thiết.
def denoise_image(image, method='bilateral'):
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    # Các phương pháp khác như đã nêu trước đây
    return image
#4. Wavelet Denoising sử dụng phép biến đổi wavelet để tách nhiễu
# và giữ lại các chi tiết quan trọng trong ảnh.
def wavelet_denoising(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Set approximation coefficients to zero
    return pywt.waverec2(coeffs_H, wavelet)
# 5.RBF Denoising sử dụng một hàm cơ sở bán kính (RBF) để điều chỉnh mức độ lọc trong mỗi vùng của ảnh.
# Đây là một phương pháp mạnh mẽ cho các loại nhiễu phức tạp.
def rbf_denoising(image, gamma=1.0):
    # Chuyển ảnh thành mảng 2D
    data = image.reshape((-1, 1))
    kernel_matrix = rbf_kernel(data, gamma=gamma)
    smoothed_data = np.dot(kernel_matrix, data) / kernel_matrix.sum(axis=1)[:, None]
    return smoothed_data.reshape(image.shape)
#6. Một phương pháp Wiener nâng cao cho phép điều chỉnh bộ lọc theo đặc trưng của ảnh,
# giúp giảm nhiễu mà không làm mất chi tiết quan trọng.

def adaptive_wiener_filter(image, window_size=5, noise_var=0.1):
    padded_image = np.pad(image, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                          mode='symmetric')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            mean = np.mean(window)
            variance = np.var(window)
            noise_est = variance - noise_var
            weight = max(0, noise_est / variance)
            filtered_image[i, j] = mean + weight * (image[i, j] - mean)

    return filtered_image
#7.Kalman Filter có thể được sử dụng trong xử lý ảnh để giảm nhiễu khi ảnh có chứa nhiễu tần số cao,
# đặc biệt là trong trường hợp theo dõi đối tượng qua nhiều khung hình.
def kalman_filter(image, process_noise=1e-4, measurement_noise=1e-1):
    kf = cv2.KalmanFilter(4, 2)  # Số trạng thái = 4, số quan sát = 2
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    filtered_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            measurement = np.array([[image[i, j]]], dtype=np.float32)
            kf.correct(measurement)
            prediction = kf.predict()
            filtered_image[i, j] = prediction[0]

    return filtered_image
#8.Dùng phép biến đổi Fourier để lọc nhiễu tần số cao trong ảnh. Sau khi biến đổi Fourier,
# bạn có thể loại bỏ tần số cao không mong muốn (nhiễu) và khôi phục lại ảnh.
def fourier_denoising(image, threshold=10):
    # Chuyển ảnh sang không gian tần số
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Tạo mask loại bỏ tần số cao
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 0

    # Chuyển ngược lại không gian không gian hình học
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)
#9.Bộ lọc Laplace cục bộ giúp làm mượt ảnh trong khi bảo tồn các chi tiết và cạnh sắc nét.
# Đây là một phương pháp xử lý ảnh mạnh mẽ cho các ứng dụng nhiễu ảnh phức tạp.
def local_laplacian_filter(image, sigma=2.0):
    # Tính toán gradient Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Thực hiện làm mượt trên ảnh gốc
    smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)

    # Trả lại ảnh với các chi tiết sắc nét
    return smoothed_image + laplacian
#10.Homomorphic filtering kết hợp giữa lọc tần số cao và tần số thấp để làm sáng bề mặt và giảm nhiễu.
# Phương pháp này thường được sử dụng trong xử lý ảnh y tế hoặc ảnh có ánh sáng không đồng đều.

def homomorphic_filter(image, cutoff=30):
    # Chuyển ảnh về không gian logarit để xử lý tần số thấp và cao
    image_log = np.log1p(np.float32(image))

    # Thực hiện phép biến đổi Fourier
    f = np.fft.fft2(image_log)
    fshift = np.fft.fftshift(f)

    # Tạo filter với cutoff frequency
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    r = np.sqrt((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol) ** 2)
    mask[r < cutoff] = 0

    # Áp dụng filter
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)

    # Quay lại không gian ảnh và chuyển về giá trị không logarit
    img_back = np.exp(np.fft.ifft2(f_ishift)) - 1
    return np.uint8(img_back)
#11.Phương pháp này là một dạng của bộ lọc trung bình, trong đó thay vì tính trung bình của một vùng,
# bạn sử dụng một median được tính từ các median của các nhóm nhỏ hơn trong vùng.
def median_of_medians(image, window_size=5):
    from scipy.ndimage import uniform_filter
    # Tạo ảnh con bằng cách cắt ảnh thành các phần nhỏ hơn
    filtered_image = np.copy(image)
    for i in range(image.shape[0] - window_size + 1):
        for j in range(image.shape[1] - window_size + 1):
            patch = image[i:i+window_size, j:j+window_size]
            filtered_image[i+window_size//2, j+window_size//2] = np.median(patch)
    return filtered_image
#12.Lọc TV là phương pháp dùng để làm mượt ảnh trong khi bảo toàn các cạnh.
# Đây là phương pháp mạnh mẽ trong các ứng dụng tái tạo ảnh và giảm nhiễu.
def total_variation_denoising(image, weight=0.1):
    # Tính toán gradient (hiệu giữa các pixel xung quanh)
    grad_x = np.diff(image, axis=0)
    grad_y = np.diff(image, axis=1)

    # Tính toán tổng biến thiên
    grad_x[grad_x > weight] = weight
    grad_y[grad_y > weight] = weight
    return grad_x, grad_y
#13.PCA có thể được áp dụng để
# loại bỏ nhiễu bằng cách giữ lại các thành phần chính trong không gian đặc trưng của ảnh.

def pca_denoising(image, n_components=50):
    # Chuyển ảnh thành vector
    reshaped_image = image.reshape((-1, 1))

    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    pca.fit(reshaped_image)

    # Chuyển đổi ảnh về không gian PCA
    transformed_image = pca.transform(reshaped_image)
    reconstructed_image = pca.inverse_transform(transformed_image)

    return reconstructed_image.reshape(image.shape)
#14.Non-Linear Diffusion, còn gọi là Perona-Malik Diffusion,
# là một phương pháp lọc mạnh mẽ cho việc giảm nhiễu mà vẫn bảo vệ các cạnh trong ảnh.

def perona_malik_diffusion(image, n_iter=10, kappa=50, gamma=0.1):
    # Khởi tạo ảnh
    img = np.float32(image)
    for _ in range(n_iter):
        gradient = np.gradient(img)
        grad_north, grad_south = gradient[0], gradient[1]
        grad_east, grad_west = gradient[2], gradient[3]

        # Tính toán các hệ số conduction
        conduction_north = np.exp(-(grad_north / kappa) ** 2)
        conduction_south = np.exp(-(grad_south / kappa) ** 2)
        conduction_east = np.exp(-(grad_east / kappa) ** 2)
        conduction_west = np.exp(-(grad_west / kappa) ** 2)

        # Cập nhật ảnh
        img += gamma * (conduction_north * grad_north + conduction_south * grad_south +
                        conduction_east * grad_east + conduction_west * grad_west)

    return np.uint8(img)
#15.Phương pháp này áp dụng biến đổi wavelet để làm mượt ảnh và
# giảm nhiễu bằng cách sử dụng một mức ngưỡng để loại bỏ các thành phần nhiễu tần số cao.
def wavelet_thresholding(image, threshold=0.5):
    coeffs = pywt.wavedec2(image, 'db1')
    coeffs_thresholded = [pywt.threshold(i, threshold, mode='soft') for i in coeffs]
    return pywt.waverec2(coeffs_thresholded, 'db1')

#16.Bilateral Grid là một phương pháp lọc hiệu quả để loại bỏ nhiễu mà vẫn bảo vệ các cạnh sắc nét trong ảnh.
# Phương pháp này sử dụng
# một lưới đối xứng để lưu trữ các giá trị pixel và sau đó tính toán lại các giá trị pixel để loại bỏ nhiễu
def bilateral_grid_denoising(image, diameter=5, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

#17.EMA có thể sử dụng để làm mượt ảnh theo thời gian,
# đặc biệt là trong các video hoặc chuỗi ảnh, làm giảm nhiễu trong mỗi khung hình.
def ema_denoising(image, alpha=0.1):
    return cv2.addWeighted(image, 1 - alpha, image, 0, alpha)
#18.K-means có thể được sử dụng để phân nhóm các giá trị pixel trong ảnh thành các cụm và
# làm mượt các giá trị của mỗi cụm, giúp giảm nhiễu.
def kmeans_denoising(image, n_clusters=2):
    reshaped_image = image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped_image)
    clustered_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
    return np.uint8(clustered_image)
#19.Sử dụng mô hình sparse representation để tái tạo ảnh từ các đặc trưng của nó,
# giúp giảm nhiễu mà không làm mất các chi tiết quan trọng.
def sparse_representation_denoising(image, n_components=100):
    # Chuyển đổi ảnh thành các đặc trưng vector
    reshaped_image = image.reshape((-1, 1))

    # Áp dụng Sparse Coding
    dictionary = DictionaryLearning(n_components=n_components, random_state=0)
    dictionary.fit(reshaped_image)

    # Xây dựng lại ảnh từ các đặc trưng đã học
    sparse_image = dictionary.inverse_transform(dictionary.transform(reshaped_image))
    return sparse_image.reshape(image.shape)

#20.Bilateral grid là một kỹ thuật cải tiến của bộ lọc bilateral,
# giúp tăng tốc quá trình lọc trong khi vẫn bảo vệ các cạnh.
def bilateral_grid_denoising(image, grid_size=5):
    # Xây dựng grid với các giá trị pixel
    grid = np.zeros_like(image)

    # Áp dụng bộ lọc bilateral vào grid
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grid[i, j] = cv2.bilateralFilter(image[i:i + grid_size, j:j + grid_size], 5, 75, 75)

    return grid

#21.Đây là một phương pháp mạnh mẽ trong việc giảm nhiễu ảnh trong khi bảo vệ các chi tiết sắc nét,
# đặc biệt hữu ích với các ảnh có nhiễu mạnh.
def tv_denoising_soft_threshold(image, weight=0.1):
    gradient_x = np.gradient(image, axis=0)
    gradient_y = np.gradient(image, axis=1)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    soft_thresholded_gradient = np.sign(gradient_magnitude) * np.maximum(gradient_magnitude - weight, 0)

    denoised_image = image - soft_thresholded_gradient
    return np.uint8(denoised_image)
#22.Local Histogram Equalization (LHE) có thể giúp giảm nhiễu
# và cải thiện độ tương phản cho các vùng ảnh có cường độ sáng tối đồng đều.

def local_hist_eq(image, kernel_size=8):
    return cv2.equalizeHist(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))
#23.Sử dụng Canny Edge Detection có thể giúp loại bỏ nhiễu đồng thời bảo vệ các cạnh quan trọng trong ảnh.
# Dù chủ yếu được dùng để phát hiện cạnh, nhưng nó cũng có thể hỗ trợ trong việc lọc nhiễu.

def canny_edge_denoising(image, low_threshold=100, high_threshold=200):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges
#24.BM3D là một phương pháp lọc mạnh mẽ cho phép giảm nhiễu trong ảnh mà không làm mờ các cạnh,
# sử dụng phương pháp "block matching" để tìm các khối tương tự trong ảnh và thay thế chúng.
def bm3d_denoising(image):
    return bm3d.bm3d(image, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

#25.Ngoài phương pháp "Perona-Malik", một số biến thể của
# Non-Linear Diffusion có thể được áp dụng để làm mượt ảnh trong khi giữ lại các chi tiết quan trọng như cạnh.
def anisotropic_diffusion(image, n_iter=10, kappa=50, gamma=0.1):
    img = np.float32(image)
    for _ in range(n_iter):
        gradient = np.gradient(img)
        grad_north, grad_south = gradient[0], gradient[1]
        grad_east, grad_west = gradient[2], gradient[3]

        conduction_north = np.exp(-(grad_north / kappa) ** 2)
        conduction_south = np.exp(-(grad_south / kappa) ** 2)
        conduction_east = np.exp(-(grad_east / kappa) ** 2)
        conduction_west = np.exp(-(grad_west / kappa) ** 2)

        img += gamma * (conduction_north * grad_north + conduction_south * grad_south +
                        conduction_east * grad_east + conduction_west * grad_west)

    return np.uint8(img)

#26.Wavelet shrinkage là một phương pháp giảm nhiễu sử dụng ngưỡng
# đối với các hệ số wavelet để loại bỏ nhiễu tần số cao, đồng thời giữ lại các đặc trưng chính trong ảnh.
def wavelet_shrinkage(image, wavelet='db1', threshold=0.5):
    coeffs = pywt.wavedec2(image, wavelet)
    coeffs_shrinked = [(pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else c) for c in coeffs]
    return pywt.waverec2(coeffs_shrinked, wavelet)
#27.Phương pháp này sử dụng phân tích ma trận để tái cấu trúc ảnh thành một ma trận có rank thấp,
# giảm nhiễu mà không làm mất các chi tiết chính trong ảnh.
def low_rank_matrix_approximation(image, n_components=20):
    model = NMF(n_components=n_components)
    W = model.fit_transform(image)
    H = model.components_
    return np.dot(W, H)
#28.Gaussian Process Regression là một phương pháp học máy có thể được sử dụng để tái tạo các giá trị
# ảnh từ một mô hình thống kê, giúp làm mượt và giảm nhiễu trong ảnh.

def gaussian_process_denoising(image):
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)

    # Xử lý mỗi pixel trong ảnh (có thể cần chia nhỏ cho ảnh lớn)
    denoised_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Dự đoán giá trị pixel bằng GPR
            gpr.fit([[i, j]], [image[i, j]])
            denoised_image[i, j] = gpr.predict([[i, j]])[0]

    return denoised_image
#29.Phương pháp Random Walks có thể được sử dụng để lọc nhiễu ảnh,
# đặc biệt hữu ích trong các tình huống mà nhiễu phân bố không đồng đều.
def random_walk_denoising(image, beta=0.1):
    rows, cols = image.shape
    smoothed_image = np.zeros_like(image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            smoothed_image[i, j] = np.mean(image[i - 1:i + 2, j - 1:j + 2]) * (1 - beta) + image[i, j] * beta

    return smoothed_image
#30.Phương pháp này lọc ảnh trong không gian gradient thay vì không gian pixel,
# giảm nhiễu bằng cách điều chỉnh gradient của ảnh.
def gradient_domain_denoising(image, sigma=1.0):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Lọc gradient
    gradient_x = cv2.GaussianBlur(gradient_x, (5, 5), sigma)
    gradient_y = cv2.GaussianBlur(gradient_y, (5, 5), sigma)

    # Xây dựng lại ảnh từ các gradient đã lọc
    denoised_image = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    return np.uint8(denoised_image)

#31.Guided Filter: Đây là một phương pháp lọc bảo vệ cạnh khá hiệu quả,
# áp dụng hướng dẫn từ một ảnh khác (thường là ảnh gốc) để điều chỉnh quá trình lọc.
def guided_filter(image, radius=5, eps=0.1):
    return cv2.ximgproc.guidedFilter(image, image, radius, eps)
#32.Joint Bilateral Filtering: Đây là một phương pháp lọc đồng bộ sử dụng thông tin từ một ảnh dẫn hướng
# (thường là ảnh gốc hoặc ảnh mờ) để áp dụng các phép lọc bilateral trong không gian ảnh.
def joint_bilateral_filter(image, guide, d=9, sigma_color=75, sigma_space=75):
    return cv2.ximgproc.jointBilateralFilter(guide, image, d, sigma_color, sigma_space)

#33.TV Denoising là một phương pháp lọc có thể giảm nhiễu trong ảnh mà không làm mất các cạnh sắc nét,
# thích hợp cho ảnh có nhiều nhiễu và vẫn giữ được các chi tiết quan trọng.
def tv_denoising(image, weight=0.1):
    return cv2.ximgproc.createTotalVariationFilter(image, weight).filter(image)
#34.Lọc khu biệt (hay còn gọi là lọc Perona-Malik) giúp giảm nhiễu trong khi giữ lại các chi tiết biên của ảnh,
#thông qua việc điều chỉnh hệ số khu biệt theo hướng các cạnh.
def anisotropic_diffusion(image, num_iterations=10, k=15, lambda_=0.25):
    img = image.astype(np.float32)
    for _ in range(num_iterations):
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        diff = np.exp(-(grad_mag / k)**2)
        img = img + lambda_ * (cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) * diff + cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) * diff)
    return np.uint8(np.clip(img, 0, 255))
#35. Phương pháp Retinex sử dụng lý thuyết của thị giác để tăng cường độ sáng và độ tương phản của ảnh, đồng thời giảm nhiễu.
#Retinex có thể cải thiện ảnh có độ sáng không đồng đều và có thể giúp làm mịn ảnh mà không làm mất chi tiết.
def retinex_filter(image, sigma=30):
    img = np.float32(image) + 1.0  # Tránh log(0)
    log_img = np.log(img)
    blur_img = cv2.GaussianBlur(log_img, (0, 0), sigma)
    return np.uint8(np.clip(np.exp(log_img - blur_img) * 255, 0, 255))
#36. Bộ lọc Gabor là một loại bộ lọc không gian tần số được sử dụng để phát hiện các chi tiết và kết cấu trong ảnh, đồng thời có thể giảm nhiễu.
# Bộ lọc này có thể hữu ích trong các ứng dụng phân tích kết cấu.
def gabor_filter(image, kernel_size=21, sigma=5.0, theta=0, lambda_=10.0, psi=0, gamma=1.0):
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, psi, gamma)
    return cv2.filter2D(image, cv2.CV_32F, kernel)
#37. Bộ lọc đệ quy có thể giảm nhiễu trong ảnh mà không làm mờ các chi tiết quan trọng.
# Bộ lọc này thường được sử dụng trong các hệ thống tín hiệu số.
def recursive_filter(image, alpha=0.5):
    img = image.astype(np.float32)
    for i in range(1, img.shape[0]):
        img[i] = alpha * img[i] + (1 - alpha) * img[i - 1]
    return np.uint8(np.clip(img, 0, 255))
#38. Lọc dựa trên dòng quang học (Optical Flow) có thể được sử dụng để giảm nhiễu trong các video hoặc ảnh liên tiếp,
# bằng cách phân tích chuyển động của các đối tượng trong ảnh.
def optical_flow_denoising(prev_image, next_image):
    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow
#39. Bộ lọc Lee là một loại bộ lọc thích ứng giúp giảm nhiễu trong ảnh vệ tinh, ảnh radar hoặc các ảnh có nhiễu Gaussian. Bộ lọc này hoạt động
# bằng cách tính toán giá trị trung bình của các pixel xung quanh mỗi pixel và điều chỉnh dựa trên độ lệch chuẩn.
def lee_filter(image, window_size=3):
    img = image.astype(np.float32)
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    mean = cv2.filter2D(img, -1, kernel)
    var = cv2.filter2D((img - mean)**2, -1, kernel)
    return np.uint8(np.clip(mean, 0, 255))  # Đơn giản hóa để biểu diễn Lee filter
#40. Phương pháp hiệu chỉnh gamma có thể giúp cải thiện độ sáng và độ tương phản của ảnh,
# đồng thời giảm nhiễu cho những vùng có độ sáng thấp.
def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def denoise_image(image, method='1.nlm'):
    if method == '1.nlm':  # Non-Local Means
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    elif method == '2.tv':  # Total Variation Denoising
        return cv2.ximgproc.createFastGlobalSmootherFilter(image, lambda_=0.1).filter(image)

    elif method == '3.gaussian':  # Gaussian Blur
        return cv2.GaussianBlur(image, (5, 5), 0)

    elif method == '4.median':  # Median Filter
        return cv2.medianBlur(image, 5)

    elif method == '5.bilateral':  # Bilateral Filter
        return cv2.bilateralFilter(image, 9, 75, 75)

    elif method == '6.wiener':  # Wiener Filter
        return wiener_filter(image)

    elif method == '7.anisotropic':  # Anisotropic Diffusion
        return anisotropic_diffusion(image)

    elif method == '8.wavelet':  # Wavelet Denoising
        return wavelet_denoising(image)

    elif method == '9.rbf':  # RBF Denoising
        return rbf_denoising(image)

    elif method == '10.adaptive_wiener':  # Adaptive Wiener Filter
        return adaptive_wiener_filter(image)

    elif method == '11.kalman':  # Kalman Filter
        return kalman_filter(image)

    elif method == '12.fourier':  # Fourier Transform Denoising
        return fourier_denoising(image)

    elif method == '13.laplacian':  # Local Laplacian Filtering
        return local_laplacian_filter(image)

    elif method == '14.homomorphic':  # Homomorphic Filtering
        return homomorphic_filter(image)

    elif method == '15.median_of_medians':  # Median of Medians
        return median_of_medians(image)

    elif method == '16.pca':  # PCA Denoising
        return pca_denoising(image)

    elif method == '17.perona_malik':  # Non-Linear Diffusion (Perona-Malik)
        return perona_malik_diffusion(image)

    elif method == '18.thresholding':  # Wavelet Thresholding
        return wavelet_thresholding(image)

    elif method == '19.bilateral_grid':  # Bilateral Grid Denoising
        return bilateral_grid_denoising(image)

    elif method == '20.ema':  # Exponential Moving Average Denoising
        return ema_denoising(image)

    elif method == '21.kmeans':  # K-means Clustering for Denoising
        return kmeans_denoising(image)

    elif method == '22.sparse_rep':  # Sparse Representation Denoising
        return sparse_representation_denoising(image)

    elif method == '23.canny_edge':  # Canny Edge Detection for Noise Reduction
        return canny_edge_denoising(image)

    elif method == '24.bm3d':  # BM3D Denoising
        return bm3d_denoising(image)

    elif method == '25.gpr':  # Gaussian Process Regression Denoising
        return gaussian_process_denoising(image)

    elif method == '26.low_rank':  # Low Rank Matrix Approximation
        return low_rank_matrix_approximation(image)

    elif method == '27.random_walk':  # Random Walks for Denoising
        return random_walk_denoising(image)
    elif method == '28.gradient_domain':  # Random Walks for Denoising
        return gradient_domain_denoising(image, sigma=1.0)

    return image  # Nếu không có phương pháp nào phù hợp, trả về ảnh gốc
