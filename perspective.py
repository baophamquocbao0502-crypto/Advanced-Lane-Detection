# =============================================================================
# Module: perspective.py
# Muc dich: Perspective Transform — Chuyen goc nhin sang Birds-Eye View
#
# Tai sao can Perspective Transform?
# - Anh tu camera xe nhin ve phia truoc -> lan duong hoi tu ve diem xa
# - 2 lan duong song song trong nhu gap nhau o duong chan troi
# - Polynomial fit se bi sai neu lam tren anh goc nhin binh thuong
#
# Sau Perspective Transform:
# - Goc nhin tu TREN XUONG (Birds-Eye View)
# - 2 lan duong song song -> THUC SU song song trong anh
# - De dang fit polynomial va tinh do cong chinh xac
#
# Nguyen ly:
# - Chon 4 diem SRC tren anh goc (hinh thang — vung duong phia truoc)
# - Chon 4 diem DST tuong ung tren anh dich (hinh chu nhat)
# - OpenCV tinh Homography Matrix M de map SRC -> DST
# - Dung M de warp anh -> Birds-Eye View
# - Luu Minv (inverse) de warp nguoc lai ket qua len anh goc
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =============================================================================
# HAM 1: get_perspective_points()
# =============================================================================
def get_perspective_points(img_shape, src_points=None, dst_points=None):
    h, w = img_shape[:2]

    if src_points is None:
        # Tọa độ tương đối → hoạt động với mọi kích thước video
        # S0/S1: đường chân trời (y ≈ 67% chiều cao)
        # S2/S3: gần xe (y ≈ 94% chiều cao)
        # S3/S0 bên trái bắt đầu sát vạch vàng, không lấy lề đường
        src = np.float32([
            [w * 0.360, h * 0.667],   # S0: trên trái  (~461, 480 cho 1280×720) — trái vạch vàng
            [w * 0.625, h * 0.667],   # S1: trên phải  (~800, 480)
            [w * 0.820, h * 0.944],   # S2: dưới phải  (~1050, 680)
            [w * 0.178, h * 0.944],   # S3: dưới trái  (~228, 680) — sát mép trái vạch vàng
        ])
    else:
        src = np.float32(src_points)

    if dst_points is None:
        # DST: hình chữ nhật với lề 20% mỗi bên để thu hẹp birds-eye view
        margin = int(0.20 * w)
        dst = np.float32([
            [margin,     0],
            [w - margin, 0],
            [w - margin, h],
            [margin,     h],
        ])
    else:
        dst = np.float32(dst_points)

    return src, dst


# =============================================================================
# HAM 2: compute_perspective_transform()
# =============================================================================
def compute_perspective_transform(src, dst):
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# =============================================================================
# HAM 3: warp_image()
# =============================================================================
def warp_image(img, M):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped


# =============================================================================
# HAM 4: unwarp_image()
# =============================================================================
def unwarp_image(warped_img, Minv, original_shape):
    h, w = original_shape[:2]
    unwarped = cv2.warpPerspective(
        warped_img, Minv, (w, h),
        flags=cv2.INTER_LINEAR
    )
    return unwarped


# =============================================================================
# HAM 5: draw_perspective_points()
# =============================================================================
def draw_perspective_points(img, src, dst=None):
    out = img.copy()
    src_int = src.astype(np.int32)
    cv2.polylines(out, [src_int.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 0, 255), thickness=3)
    for i, pt in enumerate(src_int):
        cv2.circle(out, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(out, f'S{i}', tuple(pt + [10, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# =============================================================================
# HAM 6: visualize_perspective()
# =============================================================================
def visualize_perspective(original, warped, src, save_path=None):
    orig_with_pts = draw_perspective_points(original, src)
    orig_rgb = cv2.cvtColor(orig_with_pts, cv2.COLOR_BGR2RGB)

    if len(warped.shape) == 2:
        warped_display = warped
        cmap = 'gray'
    else:
        warped_display = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        cmap = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Perspective Transform — Birds-Eye View', fontsize=15)
    axes[0].imshow(orig_rgb)
    axes[0].set_title('Original (red = SRC region)', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(warped_display, cmap=cmap)
    axes[1].set_title('After Perspective Transform (Birds-Eye View)', fontsize=12)
    axes[1].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved to: {save_path}")

    plt.show()


# =============================================================================
# CHAY THU MODULE NAY TRUC TIEP
# Lenh: python src/perspective.py
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  PERSPECTIVE TRANSFORM MODULE - AdvancedLaneDetection")
    print("=" * 60)

    import sys
    sys.path.insert(0, '.')
    from src.calibration  import load_calibration, undistort_image
    from src.thresholding import combined_threshold, region_of_interest

    TEST_IMG = "data/test_images/video_frame.jpg"
    print("\n[STEP 1] Load calibration...")
    mtx, dist = load_calibration('calibration_data.pkl')
    if mtx is None:
        print("[ERROR] Run src/calibration.py first!")
        exit()

    img = cv2.imread(TEST_IMG)
    if img is None:
        print(f"[ERROR] Cannot find: {TEST_IMG}")
        exit()

    undistorted = undistort_image(img, mtx, dist)
    print(f"  Image size: {undistorted.shape[1]}x{undistorted.shape[0]}")

    print("\n[STEP 2] Computing perspective points...")
    src, dst = get_perspective_points(undistorted.shape)
    print(f"  SRC: {src.tolist()}")
    print(f"  DST: {dst.tolist()}")

    print("\n[STEP 3] Computing homography...")
    M, Minv = compute_perspective_transform(src, dst)

    print("\n[STEP 4] Warping color image...")
    warped_color = warp_image(undistorted, M)
    os.makedirs("output/images", exist_ok=True)
    visualize_perspective(
        undistorted, warped_color, src,
        save_path="output/images/perspective_color.png"
    )

    print("\n[STEP 5] Warping binary image...")
    binary     = combined_threshold(undistorted)
    binary_roi = region_of_interest(binary)
    warped_binary = warp_image(binary_roi, M)
    visualize_perspective(
        undistorted, warped_binary, src,
        save_path="output/images/perspective_binary.png"
    )

    print("\n[STEP 6] Testing unwarp...")
    unwarped = unwarp_image(warped_color, Minv, undistorted.shape)
    cv2.imwrite("output/images/unwarped_test.png", unwarped)

    print("\n[DONE]")
    print("  -> output/images/perspective_color.png")
    print("  -> output/images/perspective_binary.png")
    print("\n[HOW TO TUNE SRC if lanes not parallel]:")
    print("  S0/S1 too high -> increase y (e.g. 450 -> 470)")
    print("  S0/S1 too low  -> decrease y (e.g. 450 -> 430)")
    print("  Too narrow     -> increase x spread of S2/S3")
    print("  Too wide       -> decrease x spread of S2/S3")