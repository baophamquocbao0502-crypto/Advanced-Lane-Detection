# =============================================================================
# Module: lane_detection.py
# Mục đích: Phát hiện làn đường và fit polynomial
#
# Tại sao cần module này?
# - Sau perspective transform, ta có Binary Image từ góc nhìn trên xuống
# - Cần xác định chính xác VỊ TRÍ của 2 làn đường (trái và phải)
# - Fit polynomial bậc 2 để biểu diễn đường cong của làn
# - Smooth kết quả giữa các frame để tránh nhảy
#
# Pipeline của module này:
#   1. Histogram → tìm vị trí ban đầu của 2 làn
#   2. Sliding Window → tìm toàn bộ pixels của mỗi làn
#   3. Polynomial Fit → fit đường cong qua các pixels
#   4. EMA Smoothing → làm mượt kết quả giữa các frame
#
# Liên hệ với code gốc của bạn:
#   - Code gốc dùng contour + fitLine → chỉ fit đường THẲNG
#   - Module này dùng Sliding Window + polyfit → fit đường CONG bậc 2
#   - Thêm EMA smoothing giống code gốc của bạn
#   - Kết quả chính xác hơn nhiều cho đường cong
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =============================================================================
# CLASS: LaneTracker
# Mục đích: Lưu trạng thái của một làn đường qua các frame
#
# Tại sao cần class này?
# - Mỗi frame xử lý độc lập → kết quả nhảy liên tục
# - LaneTracker lưu lịch sử N frame gần nhất
# - Tính trung bình → kết quả mượt hơn
# - Biết frame trước đã detect được chưa → dùng chiến lược khác nhau
# =============================================================================
class LaneTracker:
    """
    Theo dõi trạng thái của một làn đường (trái hoặc phải) qua các frame.
    """

    def __init__(self, n_frames=5, ema_alpha=0.25):
        """
        Tham số:
            n_frames  : Số frame gần nhất để tính trung bình (smoothing)
            ema_alpha : Hệ số EMA (0.0~1.0)
                        Nhỏ → mượt hơn nhưng chậm phản ứng
                        Lớn → phản ứng nhanh nhưng ít mượt
        """
        # Trạng thái detect
        self.detected = False           # Frame trước có detect được không?

        # Lưu lịch sử polynomial coefficients
        self.recent_fits   = []         # Danh sách N fits gần nhất
        self.best_fit      = None       # Trung bình của recent_fits
        self.current_fit   = None       # Fit của frame hiện tại

        # EMA smoothing (giống code gốc của bạn)
        self.ema_alpha     = ema_alpha
        self.ema_fit       = None       # Fit được smooth bằng EMA

        # Thống kê
        self.n_frames      = n_frames
        self.missed_frames = 0          # Số frame liên tiếp không detect được

    def update(self, fit):
        """
        Cập nhật tracker với polynomial fit mới.

        Tham số:
            fit : numpy array (3,) — coefficients [A, B, C]
                  của polynomial x = Ay² + By + C
                  None nếu frame này không detect được
        """
        if fit is None:
            # Không detect được frame này
            self.detected = False
            self.missed_frames += 1
            return

        # Detect thành công
        self.detected      = True
        self.missed_frames = 0
        self.current_fit   = fit

        # Thêm vào lịch sử, giữ tối đa n_frames
        self.recent_fits.append(fit)
        if len(self.recent_fits) > self.n_frames:
            self.recent_fits.pop(0)

        # Tính trung bình của N fits gần nhất
        self.best_fit = np.mean(self.recent_fits, axis=0)

        # EMA update (giống ema_update() trong code gốc của bạn)
        if self.ema_fit is None:
            self.ema_fit = fit
        else:
            self.ema_fit = (
                (1 - self.ema_alpha) * self.ema_fit +
                self.ema_alpha * fit
            )

    def get_fit(self, method='ema'):
        """
        Lấy polynomial fit đã được smooth.

        Tham số:
            method : 'ema'  → dùng EMA smoothing (mượt nhất)
                     'avg'  → dùng trung bình N frames
                     'current' → dùng fit của frame hiện tại (không smooth)

        Trả về:
            numpy array (3,) hoặc None nếu chưa có dữ liệu
        """
        if method == 'ema':
            return self.ema_fit
        elif method == 'avg':
            return self.best_fit
        else:
            return self.current_fit

    def is_lost(self, max_missed=5):
        """
        Kiểm tra xem đã mất track quá lâu chưa.

        Tham số:
            max_missed : Số frame tối đa được phép miss liên tiếp

        Trả về:
            True nếu đã mất track quá lâu → cần reset
        """
        return self.missed_frames >= max_missed

    def reset(self):
        """Reset tracker về trạng thái ban đầu."""
        self.detected      = False
        self.recent_fits   = []
        self.best_fit      = None
        self.current_fit   = None
        self.ema_fit       = None
        self.missed_frames = 0


# =============================================================================
# HÀM 1: find_lane_base()
# Mục đích: Dùng Histogram để tìm vị trí ban đầu của 2 làn đường
#
# Nguyên lý Histogram:
# - Đếm số pixels trắng theo từng cột (axis=0)
# - Cột có nhiều pixels trắng nhất = vị trí làn đường
# - Chia ảnh làm 2 nửa: nửa trái → làn trái, nửa phải → làn phải
# =============================================================================
def find_lane_base(binary_warped):
    """
    Tìm vị trí x ban đầu của 2 làn đường bằng histogram.

    Tham số:
        binary_warped : Binary image đã warp (Birds-Eye View)
                        Giá trị pixels: 0 hoặc 255

    Trả về:
        leftx_base  : Vị trí x của làn trái (pixel)
        rightx_base : Vị trí x của làn phải (pixel)
        histogram   : Histogram để debug
    """
    h, w = binary_warped.shape[:2]

    # Chỉ dùng nửa dưới của ảnh để tính histogram
    # Nửa dưới = phần đường gần xe nhất → ít nhiễu hơn
    bottom_half = binary_warped[h // 2:, :]

    # Đếm pixels trắng theo từng cột
    histogram = np.sum(bottom_half, axis=0)

    # Chia làm 2 nửa
    midpoint = w // 2

    # Tìm đỉnh cao nhất ở mỗi nửa = vị trí làn đường
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base, histogram


# =============================================================================
# HÀM 2: sliding_window_search()
# Mục đích: Tìm TẤT CẢ pixels của làn đường bằng Sliding Window
#
# Nguyên lý Sliding Window:
# - Chia ảnh thành N cửa sổ nhỏ từ DƯỚI lên TRÊN
# - Mỗi cửa sổ tìm pixels trắng trong vùng của nó
# - Cửa sổ tiếp theo dịch chuyển theo tâm của pixels tìm được
# - Giống như "leo thang" từ dưới lên trên theo làn đường
# =============================================================================
def sliding_window_search(binary_warped,
                          leftx_base, rightx_base,
                          n_windows=9,
                          margin=100,
                          minpix=50):
    """
    Tìm lane pixels bằng Sliding Window từ dưới lên trên.

    Tham số:
        binary_warped : Binary image đã warp
        leftx_base    : Vị trí x ban đầu của làn trái (từ histogram)
        rightx_base   : Vị trí x ban đầu của làn phải
        n_windows     : Số cửa sổ chia theo chiều dọc (mặc định 9)
        margin        : Nửa chiều rộng của mỗi cửa sổ (pixels)
        minpix        : Số pixels tối thiểu để dịch chuyển cửa sổ

    Trả về:
        leftx, lefty   : Tọa độ x, y của pixels làn trái
        rightx, righty : Tọa độ x, y của pixels làn phải
        out_img        : Ảnh debug với cửa sổ được vẽ
    """
    h, w = binary_warped.shape[:2]

    # Tạo ảnh output để debug (vẽ cửa sổ lên đó)
    out_img = np.dstack([binary_warped, binary_warped, binary_warped])

    # Tìm tất cả pixels trắng trong ảnh
    nonzero  = binary_warped.nonzero()   # Trả về (rows, cols) của pixels != 0
    nonzeroy = np.array(nonzero[0])      # Tọa độ y
    nonzerox = np.array(nonzero[1])      # Tọa độ x

    # Vị trí hiện tại của cửa sổ (sẽ dịch chuyển khi leo lên)
    leftx_current  = leftx_base
    rightx_current = rightx_base

    # Danh sách lưu indices của pixels trong từng cửa sổ
    left_lane_inds  = []
    right_lane_inds = []

    # Chiều cao của mỗi cửa sổ
    window_height = h // n_windows

    # ------------------------------------------------------------------
    # Sliding Window — lặp từ cửa sổ dưới cùng lên trên cùng
    # ------------------------------------------------------------------
    for window in range(n_windows):

        # Tọa độ y của cửa sổ hiện tại
        win_y_low  = h - (window + 1) * window_height   # Cạnh dưới
        win_y_high = h - window * window_height          # Cạnh trên

        # Tọa độ x của cửa sổ làn trái
        win_xleft_low  = leftx_current - margin
        win_xleft_high = leftx_current + margin

        # Tọa độ x của cửa sổ làn phải
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Vẽ cửa sổ lên ảnh debug
        cv2.rectangle(out_img,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)   # Màu xanh lá cho làn trái

        cv2.rectangle(out_img,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 0, 255), 2)   # Màu đỏ cho làn phải

        # Tìm pixels trong cửa sổ làn TRÁI
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        # Tìm pixels trong cửa sổ làn PHẢI
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Lưu indices của pixels tìm được
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Nếu tìm được đủ pixels → dịch chuyển cửa sổ theo tâm của chúng
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Ghép tất cả indices lại
    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Lấy tọa độ thực tế của pixels làn đường
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Tô màu pixels làn đường trên ảnh debug
    out_img[lefty,  leftx]  = [255, 0, 0]    # Xanh dương = làn trái
    out_img[righty, rightx] = [0, 0, 255]    # Đỏ = làn phải

    return leftx, lefty, rightx, righty, out_img


# =============================================================================
# HÀM 3: search_around_poly()
# Mục đích: Tìm lane pixels quanh polynomial đã biết từ frame trước
#
# Tại sao cần hàm này?
# - Sliding Window tốn nhiều compute (phải search toàn bộ ảnh)
# - Nếu frame trước đã detect được → làn đường ở gần vị trí cũ
# - Chỉ cần search trong vùng hẹp quanh polynomial cũ
# - Nhanh hơn nhiều → phù hợp cho video real-time
# =============================================================================
def search_around_poly(binary_warped, left_fit, right_fit, margin=80):
    """
    Tìm lane pixels trong vùng hẹp quanh polynomial đã biết.

    Dùng khi: frame trước đã detect được (detected = True)
    Nhanh hơn sliding_window_search()

    Tham số:
        binary_warped : Binary image đã warp
        left_fit      : Polynomial coefficients của làn trái [A,B,C]
        right_fit     : Polynomial coefficients của làn phải [A,B,C]
        margin        : Nửa chiều rộng vùng search (pixels)

    Trả về:
        leftx, lefty   : Tọa độ pixels làn trái
        rightx, righty : Tọa độ pixels làn phải
        out_img        : Ảnh debug
    """
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Tính vị trí x dự kiến của polynomial tại mỗi y
    # x = A*y² + B*y + C
    left_x_pred  = (left_fit[0]  * nonzeroy**2 +
                    left_fit[1]  * nonzeroy +
                    left_fit[2])

    right_x_pred = (right_fit[0] * nonzeroy**2 +
                    right_fit[1] * nonzeroy +
                    right_fit[2])

    # Chỉ giữ pixels trong vùng margin quanh polynomial
    left_lane_inds  = (nonzerox > (left_x_pred  - margin)) & \
                      (nonzerox < (left_x_pred  + margin))

    right_lane_inds = (nonzerox > (right_x_pred - margin)) & \
                      (nonzerox < (right_x_pred + margin))

    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Tạo ảnh debug
    out_img = np.dstack([binary_warped, binary_warped, binary_warped])
    out_img[lefty,  leftx]  = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img


# =============================================================================
# HÀM 4: fit_polynomial()
# Mục đích: Fit polynomial bậc 2 qua các pixels làn đường
#
# Polynomial: x = Ay² + By + C
# Tại sao fit x theo y (không phải y theo x)?
# - Làn đường gần như dọc → nhiều giá trị y cho mỗi x
# - Fit y theo x sẽ bị lỗi với đường thẳng đứng
# - Fit x theo y → không có vấn đề này
# =============================================================================
def fit_polynomial(leftx, lefty, rightx, righty, min_pixels=50):
    """
    Fit polynomial bậc 2 cho làn trái và phải.

    Tham số:
        leftx, lefty   : Tọa độ pixels làn trái
        rightx, righty : Tọa độ pixels làn phải
        min_pixels     : Số pixels tối thiểu để fit (tránh fit sai)

    Trả về:
        left_fit  : [A, B, C] coefficients làn trái  (None nếu thất bại)
        right_fit : [A, B, C] coefficients làn phải (None nếu thất bại)
    """
    left_fit  = None
    right_fit = None

    # Fit làn trái
    if len(leftx) >= min_pixels:
        # np.polyfit(y, x, 2) → fit x = Ay² + By + C
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        print(f"  [WARN] Làn trái: chỉ có {len(leftx)} pixels (cần >={min_pixels})")

    # Fit làn phải
    if len(rightx) >= min_pixels:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        print(f"  [WARN] Làn phải: chỉ có {len(rightx)} pixels (cần >={min_pixels})")

    return left_fit, right_fit


# =============================================================================
# HÀM 5: detect_lanes()
# Mục đích: Hàm CHÍNH — kết hợp tất cả bước detect làn đường
#
# Đây là hàm được gọi trong pipeline.py cho mỗi frame video
# Tự động chọn Sliding Window hoặc Search Around Poly tùy trạng thái
# =============================================================================
def detect_lanes(binary_warped, left_tracker, right_tracker):
    """
    Phát hiện làn đường trong một frame.

    Tự động chọn chiến lược:
    - Sliding Window: nếu chưa có dữ liệu từ frame trước
    - Search Around Poly: nếu đã detect được frame trước (nhanh hơn)

    Tham số:
        binary_warped  : Binary image đã warp (Birds-Eye View)
        left_tracker   : LaneTracker cho làn trái
        right_tracker  : LaneTracker cho làn phải

    Trả về:
        left_fit   : Polynomial fit làn trái (đã smooth)
        right_fit  : Polynomial fit làn phải (đã smooth)
        out_img    : Ảnh debug
    """
    # ------------------------------------------------------------------
    # Chọn chiến lược tìm kiếm
    # ------------------------------------------------------------------
    use_sliding = (
        not left_tracker.detected or
        not right_tracker.detected or
        left_tracker.is_lost() or
        right_tracker.is_lost()
    )

    if use_sliding:
        # SLIDING WINDOW: search toàn bộ ảnh
        leftx_base, rightx_base, histogram = find_lane_base(binary_warped)
        leftx, lefty, rightx, righty, out_img = sliding_window_search(
            binary_warped, leftx_base, rightx_base
        )
        method = "Sliding Window"
    else:
        # SEARCH AROUND POLY: search quanh polynomial cũ (nhanh hơn)
        prev_left  = left_tracker.get_fit()
        prev_right = right_tracker.get_fit()
        leftx, lefty, rightx, righty, out_img = search_around_poly(
            binary_warped, prev_left, prev_right
        )
        method = "Search Around Poly"

    # ------------------------------------------------------------------
    # Fit polynomial
    # Sliding Window tìm toàn ảnh → cần nhiều pixels (50)
    # Search Around Poly chỉ tìm vùng hẹp → chấp nhận ít pixels hơn (20)
    # ------------------------------------------------------------------
    min_pix = 50 if use_sliding else 20
    left_fit_raw, right_fit_raw = fit_polynomial(
        leftx, lefty, rightx, righty, min_pixels=min_pix
    )

    # ------------------------------------------------------------------
    # Cập nhật tracker (smoothing)
    # ------------------------------------------------------------------
    left_tracker.update(left_fit_raw)
    right_tracker.update(right_fit_raw)

    # Lấy fit đã được smooth
    left_fit  = left_tracker.get_fit(method='ema')
    right_fit = right_tracker.get_fit(method='ema')

    return left_fit, right_fit, out_img, method


# =============================================================================
# HÀM 6: draw_lane_area()
# Mục đích: Tô màu vùng làn đường trên ảnh Birds-Eye View
# =============================================================================
def draw_lane_area(binary_warped, left_fit, right_fit):
    """
    Tô màu vùng giữa 2 làn đường trên ảnh warped.

    Tham số:
        binary_warped : Binary image warped (để lấy kích thước)
        left_fit      : Polynomial coefficients làn trái
        right_fit     : Polynomial coefficients làn phải

    Trả về:
        Ảnh màu (BGR) với vùng làn đường được tô xanh
    """
    h, w = binary_warped.shape[:2]

    # Tạo ảnh trống để vẽ
    lane_img = np.zeros((h, w, 3), dtype=np.uint8)

    if left_fit is None or right_fit is None:
        return lane_img

    # Tạo dãy y từ trên xuống dưới
    ploty = np.linspace(0, h - 1, h)

    # Tính x tương ứng cho mỗi y
    left_fitx  = left_fit[0]  * ploty**2 + left_fit[1]  * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Tạo polygon từ 2 đường làn
    pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Tô màu xanh lá vùng giữa 2 làn
    cv2.fillPoly(lane_img, np.int32([pts]), (0, 255, 0))

    # Vẽ đường làn trái màu đỏ
    left_pts = np.int32(np.column_stack([left_fitx, ploty]))
    cv2.polylines(lane_img, [left_pts], False, (255, 0, 0), thickness=15)

    # Vẽ đường làn phải màu xanh dương
    right_pts = np.int32(np.column_stack([right_fitx, ploty]))
    cv2.polylines(lane_img, [right_pts], False, (0, 0, 255), thickness=15)

    return lane_img


# =============================================================================
# HÀM 7: visualize_lane_detection()
# Mục đích: Hiển thị kết quả lane detection để debug
# =============================================================================
def visualize_lane_detection(binary_warped, out_img,
                             left_fit, right_fit,
                             save_path=None):
    """
    Hiển thị kết quả Sliding Window và polynomial fit.
    """
    h, w = binary_warped.shape[:2]
    ploty = np.linspace(0, h - 1, h)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Lane Detection — Kết quả', fontsize=15)

    # Ảnh 1: Binary warped
    axes[0].imshow(binary_warped, cmap='gray')
    axes[0].set_title('Binary Warped (input)', fontsize=11)
    axes[0].axis('off')

    # Ảnh 2: Sliding Window debug
    axes[1].imshow(out_img)
    axes[1].set_title('Sliding Window Search', fontsize=11)

    # Vẽ polynomial lên ảnh sliding window
    if left_fit is not None:
        left_fitx = (left_fit[0] * ploty**2 +
                     left_fit[1] * ploty +
                     left_fit[2])
        axes[1].plot(left_fitx, ploty, color='yellow', linewidth=3,
                     label='Fit trái')

    if right_fit is not None:
        right_fitx = (right_fit[0] * ploty**2 +
                      right_fit[1] * ploty +
                      right_fit[2])
        axes[1].plot(right_fitx, ploty, color='cyan', linewidth=3,
                     label='Fit phải')

    axes[1].legend()
    axes[1].axis('off')

    # Ảnh 3: Kết quả tô màu
    lane_img = draw_lane_area(binary_warped, left_fit, right_fit)
    axes[2].imshow(cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Vùng làn đường (Birds-Eye View)', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Đã lưu visualization vào: {save_path}")

    plt.show()


# =============================================================================
# CHẠY THỬ MODULE NÀY TRỰC TIẾP
# Khi chạy: python src/lane_detection.py
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  LANE DETECTION MODULE - AdvancedLaneDetection")
    print("=" * 60)

    import sys
    sys.path.insert(0, '.')
    from src.calibration  import load_calibration, undistort_image
    from src.thresholding import combined_threshold, region_of_interest
    from src.perspective  import (get_perspective_points,
                                  compute_perspective_transform,
                                  warp_image)

    TEST_IMG = "data/test_images/test1.jpg"

    # ------------------------------------------------------------------
    # BƯỚC 1: Load và xử lý ảnh (các bước trước)
    # ------------------------------------------------------------------
    print("\n[BƯỚC 1] Load calibration...")
    mtx, dist = load_calibration('calibration_data.pkl')
    if mtx is None:
        print("[LỖI] Chạy src/calibration.py trước!")
        exit()

    print("[BƯỚC 2] Đọc và undistort ảnh...")
    img = cv2.imread(TEST_IMG)
    if img is None:
        print(f"[LỖI] Không tìm thấy: {TEST_IMG}")
        exit()

    undistorted = undistort_image(img, mtx, dist)

    print("[BƯỚC 3] Thresholding...")
    binary = combined_threshold(undistorted)
    binary_roi = region_of_interest(binary)

    print("[BƯỚC 4] Perspective transform...")
    src, dst = get_perspective_points(undistorted.shape)
    M, Minv  = compute_perspective_transform(src, dst)
    binary_warped = warp_image(binary_roi, M)

    # ------------------------------------------------------------------
    # BƯỚC 2: Lane Detection
    # ------------------------------------------------------------------
    print("\n[BƯỚC 5] Tìm vị trí làn đường bằng Histogram...")
    leftx_base, rightx_base, histogram = find_lane_base(binary_warped)
    print(f"  Làn trái  x = {leftx_base}")
    print(f"  Làn phải  x = {rightx_base}")

    print("\n[BƯỚC 6] Sliding Window Search...")
    leftx, lefty, rightx, righty, out_img = sliding_window_search(
        binary_warped, leftx_base, rightx_base
    )
    print(f"  Pixels làn trái : {len(leftx)}")
    print(f"  Pixels làn phải : {len(rightx)}")

    print("\n[BƯỚC 7] Fit Polynomial...")
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    if left_fit is not None:
        print(f"  Left fit  [A,B,C] = {left_fit}")
    if right_fit is not None:
        print(f"  Right fit [A,B,C] = {right_fit}")

    # ------------------------------------------------------------------
    # BƯỚC 3: Test LaneTracker (smoothing)
    # ------------------------------------------------------------------
    print("\n[BƯỚC 8] Test LaneTracker...")
    left_tracker  = LaneTracker(n_frames=5, ema_alpha=0.25)
    right_tracker = LaneTracker(n_frames=5, ema_alpha=0.25)

    left_tracker.update(left_fit)
    right_tracker.update(right_fit)

    left_fit_smooth  = left_tracker.get_fit('ema')
    right_fit_smooth = right_tracker.get_fit('ema')
    print(f"  Left  EMA fit = {left_fit_smooth}")
    print(f"  Right EMA fit = {right_fit_smooth}")

    # ------------------------------------------------------------------
    # BƯỚC 4: Lưu kết quả
    # ------------------------------------------------------------------
    print("\n[BƯỚC 9] Lưu kết quả visualization...")
    os.makedirs("output/images", exist_ok=True)

    visualize_lane_detection(
        binary_warped, out_img,
        left_fit_smooth, right_fit_smooth,
        save_path="output/images/lane_detection_result.png"
    )

    print("\n[HOÀN THÀNH] File đã lưu:")
    print("  → output/images/lane_detection_result.png")


# =============================================================================
# PIPELINE CỦA TOÀN BỘ PROJECT — AdvancedLaneDetection
# =============================================================================
#
#  [VIDEO FRAME]
#       │
#       ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 1: calibration.py   ✅ XONG                                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ undistorted frame
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 2: thresholding.py  ✅ XONG                                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ binary image
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 3: perspective.py   ✅ XONG                                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ binary warped + M, Minv
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 4: lane_detection.py  ← BẠN ĐANG Ở ĐÂY                     │
#  │  - find_lane_base()     → Histogram tìm vị trí ban đầu            │
#  │  - sliding_window_search() → Tìm toàn bộ lane pixels              │
#  │  - search_around_poly() → Tìm nhanh nếu đã có fit từ trước        │
#  │  - fit_polynomial()     → x = Ay² + By + C                        │
#  │  - LaneTracker          → EMA smoothing giữa các frame            │
#  │  - detect_lanes()       → Hàm chính gọi trong pipeline            │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ left_fit, right_fit (smoothed)
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 5: curvature.py                                              │
#  │  Tính độ cong và vị trí xe                                         │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ curvature, offset
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 6: pipeline.py                                               │
#  │  Warp ngược + tô màu + xuất video                                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │
#                                 ▼
#                        [VIDEO KẾT QUẢ]
#
# =============================================================================