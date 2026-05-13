# =============================================================================
# Module: calibration.py
# Mục đích: Hiệu chỉnh camera (Camera Calibration)
#
# Tại sao cần calibration?
# - Camera thực tế luôn bị méo hình (distortion) do thấu kính
# - Méo hình làm cho đường thẳng trông bị cong trong ảnh
# - Nếu không hiệu chỉnh, tất cả các bước sau (threshold, perspective,
#   lane detection) sẽ bị sai lệch
# - Sau calibration, ảnh sẽ phản ánh đúng thực tế hơn
#
# Nguyên lý hoạt động:
# 1. Chụp nhiều ảnh bàn cờ (chessboard) từ nhiều góc độ khác nhau
# 2. Tìm các góc (corners) của bàn cờ trong ảnh → imgpoints (2D)
# 3. Biết trước tọa độ thực tế của các góc đó → objpoints (3D)
# 4. OpenCV tính ra camera matrix và distortion coefficients
# 5. Dùng các thông số đó để "undistort" mọi ảnh từ camera này
#
# Liên hệ với sách Sjafrie Chapter 2:
# - Đây chính là "Intrinsic calibration" của camera sensor
# - Camera matrix chứa focal length (fx, fy) và principal point (cx, cy)
# - Distortion coefficients gồm radial (k1,k2,k3) và tangential (p1,p2)
# =============================================================================
 
import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
 
 
# =============================================================================
# HÀM 1: calibrate_camera()
# Mục đích: Tính camera matrix và distortion coefficients từ ảnh chessboard
# =============================================================================
def calibrate_camera(calibration_dir, chessboard_size=(9, 6), show_corners=False):
    """
    Thực hiện camera calibration từ tập ảnh chessboard.
 
    Tham số:
        calibration_dir  : đường dẫn đến thư mục chứa ảnh calibration
                           (vd: 'data/camera_cal/')
        chessboard_size  : số góc nội (inner corners) của bàn cờ
                           mặc định (9, 6) — 9 cột, 6 hàng
        show_corners     : True nếu muốn hiển thị ảnh với corners được vẽ
 
    Trả về:
        ret  : True nếu calibration thành công
        mtx  : Camera matrix (3x3) — chứa thông số nội của camera
        dist : Distortion coefficients — hệ số méo hình
        rvecs: Rotation vectors — góc xoay cho mỗi ảnh calibration
        tvecs: Translation vectors — dịch chuyển cho mỗi ảnh calibration
        None nếu không tìm thấy đủ ảnh calibration
    """
 
    # ------------------------------------------------------------------
    # Bước 1: Chuẩn bị object points (tọa độ 3D thực tế của góc bàn cờ)
    # Giả sử bàn cờ nằm phẳng trên mặt phẳng z=0
    # Tọa độ sẽ là: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    # ------------------------------------------------------------------
    cols, rows = chessboard_size
 
    # Tạo lưới tọa độ 3D cho một bàn cờ
    # np.mgrid tạo ra lưới điểm theo dạng [[0,1,2...8], [0,0,0...0]]
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Ví dụ với (9,6): objp = [[0,0,0],[1,0,0],...,[8,5,0]]
 
    # ------------------------------------------------------------------
    # Bước 2: Tạo danh sách để lưu points từ tất cả ảnh
    # ------------------------------------------------------------------
    objpoints = []  # Tọa độ 3D thực tế — giống nhau cho mọi ảnh
    imgpoints = []  # Tọa độ 2D pixel — khác nhau cho mỗi ảnh
 
    # ------------------------------------------------------------------
    # Bước 3: Đọc tất cả ảnh calibration
    # ------------------------------------------------------------------
    # Tìm tất cả file .jpg trong thư mục calibration
    image_paths = glob.glob(os.path.join(calibration_dir, '*.jpg'))
    image_paths += glob.glob(os.path.join(calibration_dir, '*.png'))
 
    if len(image_paths) == 0:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh trong: {calibration_dir}")
        print("  → Hãy đặt ảnh chessboard vào thư mục data/camera_cal/")
        return None
 
    print(f"[INFO] Tìm thấy {len(image_paths)} ảnh calibration")
 
    successful = 0  # Đếm số ảnh detect corners thành công
 
    for idx, fpath in enumerate(image_paths):
        # Đọc ảnh
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [SKIP] Không đọc được: {fpath}")
            continue
 
        # Chuyển sang grayscale — cv2.findChessboardCorners cần ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # ------------------------------------------------------------------
        # Bước 4: Tìm corners trong ảnh bàn cờ
        # cv2.findChessboardCorners trả về:
        #   ret     : True nếu tìm thấy đủ số corners
        #   corners : tọa độ 2D pixel của từng corner
        # ------------------------------------------------------------------
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
 
        if ret:
            # Tìm thấy corners → thêm vào danh sách
            objpoints.append(objp)
 
            # Tinh chỉnh vị trí corners chính xác hơn đến sub-pixel
            # criteria: dừng sau 30 iteration hoặc khi thay đổi < 0.001
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners_refined)
            successful += 1
 
            # Vẽ corners lên ảnh để kiểm tra (tùy chọn)
            if show_corners:
                img_corners = img.copy()
                cv2.drawChessboardCorners(
                    img_corners, chessboard_size, corners_refined, ret
                )
                cv2.imshow(f'Corners - {os.path.basename(fpath)}', img_corners)
                cv2.waitKey(300)  # Hiển thị 300ms rồi chuyển ảnh tiếp theo
 
            print(f"  [OK] {os.path.basename(fpath)} — corners found")
        else:
            print(f"  [FAIL] {os.path.basename(fpath)} — corners NOT found")
 
    if show_corners:
        cv2.destroyAllWindows()
 
    print(f"\n[INFO] Thành công: {successful}/{len(image_paths)} ảnh")
 
    # Cần ít nhất 5 ảnh để calibration đáng tin cậy
    if successful < 5:
        print("[LỖI] Cần ít nhất 5 ảnh thành công để calibrate!")
        return None
 
    # ------------------------------------------------------------------
    # Bước 5: Tính camera matrix và distortion coefficients
    # cv2.calibrateCamera nhận:
    #   objpoints : danh sách tọa độ 3D
    #   imgpoints : danh sách tọa độ 2D tương ứng
    #   gray.shape[::-1] : kích thước ảnh (width, height)
    # Trả về:
    #   ret   : RMS re-projection error (càng nhỏ càng tốt, <1.0 là tốt)
    #   mtx   : Camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    #   dist  : Distortion coefficients [k1,k2,p1,p2,k3]
    #   rvecs : Rotation vectors
    #   tvecs : Translation vectors
    # ------------------------------------------------------------------
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
 
    print(f"\n[CALIBRATION HOÀN THÀNH]")
    print(f"  RMS Re-projection Error: {ret:.4f} (< 1.0 là tốt)")
    print(f"  Camera Matrix (mtx):\n{mtx}")
    print(f"  Distortion Coefficients: {dist.ravel()}")
 
    return ret, mtx, dist, rvecs, tvecs
 
 
# =============================================================================
# HÀM 2: save_calibration()
# Mục đích: Lưu kết quả calibration ra file để dùng lại
# → Không cần chạy calibration mỗi lần, chỉ cần load từ file
# =============================================================================
def save_calibration(mtx, dist, save_path='calibration_data.pkl'):
    """
    Lưu camera matrix và distortion coefficients ra file pickle.
 
    Tham số:
        mtx       : Camera matrix
        dist      : Distortion coefficients
        save_path : Đường dẫn file lưu (mặc định: 'calibration_data.pkl')
    """
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coefficients': dist
    }
 
    with open(save_path, 'wb') as f:
        pickle.dump(calibration_data, f)
 
    print(f"[INFO] Đã lưu calibration data vào: {save_path}")
 
 
# =============================================================================
# HÀM 3: load_calibration()
# Mục đích: Load kết quả calibration từ file đã lưu trước đó
# =============================================================================
def load_calibration(load_path='calibration_data.pkl'):
    """
    Load camera matrix và distortion coefficients từ file.
 
    Tham số:
        load_path : Đường dẫn file calibration đã lưu
 
    Trả về:
        mtx  : Camera matrix
        dist : Distortion coefficients
        None nếu file không tồn tại
    """
    if not os.path.exists(load_path):
        print(f"[LỖI] Không tìm thấy file: {load_path}")
        print("  → Hãy chạy calibrate_camera() trước")
        return None, None
 
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
 
    mtx  = data['camera_matrix']
    dist = data['dist_coefficients']
 
    print(f"[INFO] Đã load calibration từ: {load_path}")
    return mtx, dist
 
 
# =============================================================================
# HÀM 4: undistort_image()
# Mục đích: Áp dụng undistortion lên một ảnh
# → Đây là hàm sẽ được gọi trong pipeline chính cho mỗi frame video
# =============================================================================
def undistort_image(img, mtx, dist):
    """
    Loại bỏ méo hình (distortion) khỏi ảnh.
 
    Tham số:
        img  : Ảnh đầu vào (BGR format từ OpenCV)
        mtx  : Camera matrix
        dist : Distortion coefficients
 
    Trả về:
        Ảnh đã được undistort
    """
    if mtx is None or dist is None:
        print("[CẢNH BÁO] mtx hoặc dist là None — trả về ảnh gốc")
        return img
 
    # cv2.undistort là hàm chính để loại bỏ méo hình
    # Tham số None ở cuối = dùng cùng camera matrix cho output
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
 
 
# =============================================================================
# HÀM 5: visualize_undistortion()
# Mục đích: So sánh ảnh gốc vs ảnh đã undistort để kiểm tra kết quả
# =============================================================================
def visualize_undistortion(original, undistorted, save_path=None):
    """
    Hiển thị ảnh gốc và ảnh undistort cạnh nhau để so sánh.
 
    Tham số:
        original    : Ảnh gốc (BGR)
        undistorted : Ảnh sau khi undistort (BGR)
        save_path   : Nếu có, lưu ảnh so sánh vào đường dẫn này
    """
    # Chuyển BGR → RGB để matplotlib hiển thị đúng màu
    orig_rgb  = cv2.cvtColor(original,    cv2.COLOR_BGR2RGB)
    undis_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    axes[0].imshow(orig_rgb)
    axes[0].set_title('Ảnh gốc (có méo hình)', fontsize=13)
    axes[0].axis('off')
 
    axes[1].imshow(undis_rgb)
    axes[1].set_title('Ảnh sau Undistortion', fontsize=13)
    axes[1].axis('off')
 
    plt.suptitle('Camera Calibration — Kết quả Undistortion', fontsize=15)
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Đã lưu ảnh so sánh vào: {save_path}")
 
    plt.show()
 
 
# =============================================================================
# CHẠY THỬ MODULE NÀY TRỰC TIẾP
# Khi chạy: python src/calibration.py
# =============================================================================
if __name__ == "__main__":
 
    print("=" * 60)
    print("  CAMERA CALIBRATION MODULE - AdvancedLaneDetection")
    print("=" * 60)
 
    # Đường dẫn thư mục chứa ảnh chessboard
    CAL_DIR  = "data/camera_cal"
    SAVE_PKL = "calibration_data.pkl"
    TEST_IMG = "data/test_images/test1.jpg"  # Ảnh để test undistortion
 
    # ------------------------------------------------------------------
    # BƯỚC 1: Chạy calibration
    # ------------------------------------------------------------------
    print("\n[BƯỚC 1] Chạy Camera Calibration...")
    result = calibrate_camera(
        calibration_dir=CAL_DIR,
        chessboard_size=(9, 6),
        show_corners=True       # Hiển thị ảnh với corners được vẽ
    )
 
    if result is None:
        print("\n[KẾT THÚC] Calibration thất bại.")
        print("  Hướng dẫn: Đặt ảnh chessboard vào data/camera_cal/")
        print("  Tải ảnh mẫu từ: github.com/udacity/CarND-Camera-Calibration")
        exit()
 
    ret, mtx, dist, rvecs, tvecs = result
 
    # ------------------------------------------------------------------
    # BƯỚC 2: Lưu kết quả để dùng lại
    # ------------------------------------------------------------------
    print("\n[BƯỚC 2] Lưu calibration data...")
    save_calibration(mtx, dist, save_path=SAVE_PKL)
 
    # ------------------------------------------------------------------
    # BƯỚC 3: Test load lại từ file
    # ------------------------------------------------------------------
    print("\n[BƯỚC 3] Test load calibration từ file...")
    mtx_loaded, dist_loaded = load_calibration(load_path=SAVE_PKL)
 
    # ------------------------------------------------------------------
    # BƯỚC 4: Test undistortion trên ảnh thực tế
    # ------------------------------------------------------------------
    print("\n[BƯỚC 4] Test undistortion trên ảnh thực tế...")
 
    if os.path.exists(TEST_IMG):
        img = cv2.imread(TEST_IMG)
        undistorted = undistort_image(img, mtx_loaded, dist_loaded)
 
        # Lưu ảnh so sánh
        os.makedirs("output/images", exist_ok=True)
        visualize_undistortion(
            img, undistorted,
            save_path="output/images/undistortion_comparison.png"
        )
    else:
        print(f"  [SKIP] Không tìm thấy ảnh test: {TEST_IMG}")
        print("  → Đặt ảnh test vào data/test_images/test1.jpg")
 
    print("\n[HOÀN THÀNH] Module calibration.py sẵn sàng!")
 
 
# =============================================================================
# PIPELINE CỦA TOÀN BỘ PROJECT — AdvancedLaneDetection
# =============================================================================
#
#  [VIDEO FRAME]
#       │
#       ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 1: calibration.py  ← BẠN ĐANG Ở ĐÂY                        │
#  │  - Đọc ảnh chessboard từ data/camera_cal/                          │
#  │  - Tính camera matrix (mtx) và distortion coefficients (dist)      │
#  │  - Lưu vào calibration_data.pkl để dùng lại                        │
#  │  - Undistort mỗi frame video trước khi xử lý tiếp                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ mtx, dist → undistorted frame
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 2: thresholding.py                                           │
#  │  - Chuyển frame sang HLS color space                               │
#  │  - Áp dụng Sobel gradient (phát hiện cạnh)                        │
#  │  - Kết hợp thành Binary Image (0/1)                                │
#  │  - Chỉ giữ lại pixels có khả năng là làn đường                    │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ binary image
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 3: perspective.py                                            │
#  │  - Chuyển góc nhìn sang Birds-Eye View (nhìn từ trên xuống)       │
#  │  - Làn đường song song sẽ trông thực sự song song                 │
#  │  - Lưu Inverse Matrix để warp ngược lại sau này                   │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ warped binary image
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 4: lane_detection.py                                         │
#  │  - Dùng Histogram tìm vị trí ban đầu của 2 làn đường              │
#  │  - Sliding Window quét từ dưới lên trên tìm lane pixels            │
#  │  - Fit Polynomial bậc 2: x = Ay² + By + C                         │
#  │  - EMA smoothing để kết quả mượt giữa các frame                   │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ left_fit, right_fit (coefficients)
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 5: curvature.py                                              │
#  │  - Tính bán kính cong (Radius of Curvature) của làn đường         │
#  │  - Tính vị trí xe lệch bao nhiêu so với tâm làn (offset)         │
#  │  - Chuyển từ pixel sang đơn vị thực (mét)                         │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ curvature (m), offset (m)
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 6: pipeline.py                                               │
#  │  - Warp kết quả ngược lại lên ảnh gốc (Inverse Perspective)       │
#  │  - Tô màu vùng làn đường (màu xanh lá)                            │
#  │  - Hiển thị thông số: curvature, offset lên ảnh                   │
#  │  - Xuất video kết quả vào output/videos/                           │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │
#                                 ▼
#                        [VIDEO KẾT QUẢ]
#
# =============================================================================