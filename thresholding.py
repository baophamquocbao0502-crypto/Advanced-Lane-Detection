# =============================================================================
# Module: thresholding.py
# Mục đích: Tạo Binary Image để highlight làn đường
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def sobel_threshold(img_bgr, orient='x', sobel_kernel=3, thresh=(10, 100)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def magnitude_threshold(img_bgr, sobel_kernel=3, thresh=(30, 100)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def direction_threshold(img_bgr, sobel_kernel=15, thresh=(0.7, 1.3)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(direction)
    binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary


def hls_white_threshold(img_bgr, thresh_l=(150, 255), thresh_s=(10, 80)):
    """Detect vạch trắng — L=150 bắt vạch đứt đoạn, S=10-80 theo giá trị thực tế."""
    # Dùng L=150 (thay vì 200) để bắt vạch trắng đứt đoạn và bóng đổ nhẹ
    # Dùng S=10-80 (thay vì 0-60) dựa trên giá trị thực tế L=167, S=59 tại vạch trắng
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary = np.zeros(l_channel.shape, dtype=np.uint8)
    binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1]) &
           (s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
    return binary


def hls_yellow_threshold(img_bgr,
                         thresh_h=(10, 40),
                         thresh_l=(30, 200),
                         thresh_s=(60, 255)):
    """Detect vạch vàng — H trong range vàng, S cao."""
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary = np.zeros(h_channel.shape, dtype=np.uint8)
    binary[(h_channel >= thresh_h[0]) & (h_channel <= thresh_h[1]) &
           (l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1]) &
           (s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
    return binary


def combined_threshold(img_bgr,
                       use_gradient=True,
                       use_color=True,
                       morph_cleanup=True):
    """Kết hợp gradient + color threshold thành Binary Image cuối cùng."""
    h, w = img_bgr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    if use_gradient:
        gradx = sobel_threshold(img_bgr, orient='x', sobel_kernel=3, thresh=(20, 100))
        grady = sobel_threshold(img_bgr, orient='y', sobel_kernel=3, thresh=(20, 100))
        mag   = magnitude_threshold(img_bgr, sobel_kernel=3, thresh=(30, 100))
        dire  = direction_threshold(img_bgr, sobel_kernel=15, thresh=(0.7, 1.3))
        gradient_binary = np.zeros((h, w), dtype=np.uint8)
        gradient_binary[((gradx == 1) & (grady == 1)) |
                        ((mag == 1) & (dire == 1))] = 1
        combined[gradient_binary == 1] = 255

    if use_color:
        # thresh_l=120 và thresh_s=0-100 để bắt vạch trắng đứt đoạn mờ trên đường cao tốc
        white  = hls_white_threshold(img_bgr, thresh_l=(120, 255), thresh_s=(0, 100))
        yellow = hls_yellow_threshold(img_bgr, thresh_h=(15, 35),
                                               thresh_l=(30, 200),
                                               thresh_s=(80, 255))
        color_binary = np.zeros((h, w), dtype=np.uint8)
        color_binary[(white == 1) | (yellow == 1)] = 1
        combined[color_binary == 1] = 255

    if morph_cleanup:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return combined


def region_of_interest(binary_img, vertices=None):
    h, w = binary_img.shape[:2]

    if vertices is None:
        # Tọa độ tương đối → hoạt động với mọi kích thước video
        # Bên trái bắt đầu từ ~19% (sát vạch vàng, không lấy lề đường)
        vertices = np.array([[
            (int(0.15 * w), h),              # Dưới trái (~192px cho 1280) — trái vạch vàng
            (int(0.34 * w), int(0.63 * h)),  # Trên trái (~435px) — trái vạch vàng tại y=454
            (int(0.74 * w), int(0.63 * h)),  # Trên phải (~947px)
            (int(0.86 * w), h),              # Dưới phải (~1100px)
        ]], dtype=np.int32)

    mask = np.zeros_like(binary_img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(binary_img, mask)


def visualize_thresholds(img_bgr, save_path=None):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gradx    = sobel_threshold(img_bgr, orient='x')
    grady    = sobel_threshold(img_bgr, orient='y')
    mag      = magnitude_threshold(img_bgr)
    dire     = direction_threshold(img_bgr)
    white    = hls_white_threshold(img_bgr)
    yellow   = hls_yellow_threshold(img_bgr)
    combined = combined_threshold(img_bgr)
    roi      = region_of_interest(combined)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Thresholding — Kết quả từng bước', fontsize=16)

    images = [
        (img_rgb,  'Ảnh gốc (sau undistort)',  False),
        (gradx,    'Sobel X gradient',          True),
        (grady,    'Sobel Y gradient',          True),
        (mag,      'Magnitude threshold',       True),
        (dire,     'Direction threshold',       True),
        (white,    'HLS White threshold',       True),
        (yellow,   'HLS Yellow threshold',      True),
        (combined, 'Combined (tất cả kết hợp)', True),
        (roi,      'Sau ROI (vùng quan tâm)',    True),
    ]

    for ax, (img, title, is_binary) in zip(axes.flatten(), images):
        ax.imshow(img, cmap='gray' if is_binary else None)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Đã lưu visualization vào: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  THRESHOLDING MODULE - AdvancedLaneDetection")
    print("=" * 60)

    import sys
    sys.path.insert(0, '.')
    from src.calibration import load_calibration, undistort_image

    TEST_IMG = "data/test_images/video_frame.jpg"

    print("\n[BƯỚC 1] Load calibration data...")
    mtx, dist = load_calibration('calibration_data.pkl')
    if mtx is None:
        print("[LỖI] Chưa có calibration data!")
        exit()

    print("\n[BƯỚC 2] Đọc và undistort ảnh test...")
    if not os.path.exists(TEST_IMG):
        print(f"[LỖI] Không tìm thấy: {TEST_IMG}")
        exit()

    img = cv2.imread(TEST_IMG)
    undistorted = undistort_image(img, mtx, dist)

    print("\n[BƯỚC 3] Chạy Combined Threshold...")
    binary = combined_threshold(undistorted)

    print("\n[BƯỚC 4] Áp dụng Region of Interest...")
    roi = region_of_interest(binary)

    print("\n[BƯỚC 5] Lưu kết quả visualization...")
    os.makedirs("output/images", exist_ok=True)
    visualize_thresholds(undistorted,
                         save_path="output/images/thresholding_result.png")
    cv2.imwrite("output/images/binary_output.png", binary)
    cv2.imwrite("output/images/roi_output.png", roi)

    print("\n[HOÀN THÀNH]")
    print("  → output/images/thresholding_result.png")
    print("  → output/images/binary_output.png")
    print("  → output/images/roi_output.png")