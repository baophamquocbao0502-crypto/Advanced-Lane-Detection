# =============================================================================
# Module: curvature.py
# Purpose: Calculate radius of curvature and vehicle offset from lane center
#
# Formula - Radius of Curvature:
#   R = (1 + (dx/dy)^2)^(3/2) / |d^2x/dy^2|
#   With polynomial x = Ay^2 + By + C:
#   dx/dy   = 2Ay + B
#   d^2x/dy^2 = 2A
#   -> R = (1 + (2Ay + B)^2)^(3/2) / |2A|
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Conversion factors: pixels to meters
YM_PER_PIX = 30 / 720   # meters per pixel in y direction
XM_PER_PIX = 3.7 / 700  # meters per pixel in x direction


# =============================================================================
# FUNCTION 1: calculate_curvature()
# =============================================================================
def calculate_curvature(left_fit, right_fit, img_shape,
                        ym_per_pix=YM_PER_PIX,
                        xm_per_pix=XM_PER_PIX):
    """
    Calculate radius of curvature for both lane lines.

    Args:
        left_fit    : Polynomial [A,B,C] for left lane (pixel space)
        right_fit   : Polynomial [A,B,C] for right lane (pixel space)
        img_shape   : (height, width) of image
        ym_per_pix  : Meters per pixel in y direction
        xm_per_pix  : Meters per pixel in x direction

    Returns:
        left_curverad  : Left lane curvature radius (meters)
        right_curverad : Right lane curvature radius (meters)
        mean_curverad  : Average curvature radius (meters)
    """
    if left_fit is None or right_fit is None:
        return None, None, None

    h, w = img_shape[:2]
    y_eval = h - 1

    # Generate y values
    ploty = np.linspace(0, h - 1, h)

    # Calculate x values from polynomial
    left_fitx  = left_fit[0]  * ploty**2 + left_fit[1]  * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Re-fit in meter space
    left_fit_cr  = np.polyfit(ploty * ym_per_pix, left_fitx  * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate curvature at bottom of image
    y_eval_m = y_eval * ym_per_pix

    left_curverad = (
        (1 + (2 * left_fit_cr[0]  * y_eval_m + left_fit_cr[1])**2)**1.5
    ) / np.absolute(2 * left_fit_cr[0])

    right_curverad = (
        (1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1])**2)**1.5
    ) / np.absolute(2 * right_fit_cr[0])

    mean_curverad = (left_curverad + right_curverad) / 2.0

    return left_curverad, right_curverad, mean_curverad


# =============================================================================
# FUNCTION 2: calculate_vehicle_offset()
# =============================================================================
def calculate_vehicle_offset(left_fit, right_fit, img_shape,
                              xm_per_pix=XM_PER_PIX):
    """
    Calculate how far the vehicle is from the center of the lane.

    Args:
        left_fit    : Polynomial [A,B,C] for left lane
        right_fit   : Polynomial [A,B,C] for right lane
        img_shape   : (height, width) of image
        xm_per_pix  : Meters per pixel in x direction

    Returns:
        offset_m  : Offset in meters (>0 = right, <0 = left)
        offset_px : Offset in pixels
    """
    if left_fit is None or right_fit is None:
        return None, None

    h, w = img_shape[:2]
    y_bottom = h - 1

    left_x  = left_fit[0]  * y_bottom**2 + left_fit[1]  * y_bottom + left_fit[2]
    right_x = right_fit[0] * y_bottom**2 + right_fit[1] * y_bottom + right_fit[2]

    lane_center = (left_x + right_x) / 2.0
    img_center  = w / 2.0

    offset_px = img_center - lane_center
    offset_m  = offset_px * xm_per_pix

    return offset_m, offset_px


# =============================================================================
# FUNCTION 3: get_curvature_description()
# =============================================================================
def get_curvature_description(curvature_m):
    """Convert curvature value to text description."""
    if curvature_m is None:
        return "N/A"
    if curvature_m > 3000:
        return "Straight road"
    elif curvature_m > 1000:
        return f"Slight curve R={curvature_m:.0f}m"
    elif curvature_m > 500:
        return f"Moderate curve R={curvature_m:.0f}m"
    else:
        return f"Sharp curve R={curvature_m:.0f}m"


# =============================================================================
# FUNCTION 4: get_offset_description()
# =============================================================================
def get_offset_description(offset_m):
    """Convert offset value to text description."""
    if offset_m is None:
        return "N/A"
    abs_offset = abs(offset_m)
    direction  = "Right" if offset_m > 0 else "Left"
    if abs_offset < 0.05:
        return "Centered"
    else:
        return f"{direction} {abs_offset:.2f}m"


# =============================================================================
# FUNCTION 5: draw_info_on_frame()
# =============================================================================
def draw_info_on_frame(frame, curvature_m, offset_m,
                       left_curverad=None, right_curverad=None):
    """
    Draw curvature and offset information on video frame.

    Args:
        frame         : Input BGR frame
        curvature_m   : Mean curvature radius (meters)
        offset_m      : Vehicle offset (meters)
        left_curverad : Left lane curvature (optional)
        right_curverad: Right lane curvature (optional)

    Returns:
        Frame with information overlay
    """
    out = frame.copy()

    # Draw dark background for text
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (530, 185), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray  = (180, 180, 180)

    # Line 1: Curvature
    curve_text = f"Curvature: {get_curvature_description(curvature_m)}"
    cv2.putText(out, curve_text,
                (20, 55), font, 0.7, white, 2, cv2.LINE_AA)

    # Line 2: Left / Right individual curvature
    if left_curverad is not None and right_curverad is not None:
        lr_text = f"Left: {left_curverad:.0f}m   Right: {right_curverad:.0f}m"
        cv2.putText(out, lr_text,
                    (20, 92), font, 0.55, gray, 1, cv2.LINE_AA)

    # Line 3: Vehicle offset
    off_text = f"Offset: {get_offset_description(offset_m)}"
    cv2.putText(out, off_text,
                (20, 132), font, 0.7, white, 2, cv2.LINE_AA)

    # Line 4: Raw offset value
    if offset_m is not None:
        val_text = f"Value: {offset_m:+.3f}m"
        cv2.putText(out, val_text,
                    (20, 168), font, 0.5, gray, 1, cv2.LINE_AA)

    return out


# =============================================================================
# FUNCTION 6: visualize_curvature()
# =============================================================================
def visualize_curvature(img_shape, left_fit, right_fit,
                        curvature_m, offset_m, save_path=None):
    """Display polynomial curves and curvature/offset statistics."""
    h, w = img_shape[:2]
    ploty = np.linspace(0, h - 1, h)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Curvature and Vehicle Offset', fontsize=15)

    # Left plot: Polynomial curves
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)
    axes[0].set_title('Polynomial Fit (Birds-Eye View)', fontsize=12)
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    axes[0].set_facecolor('#1a1a2e')

    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        axes[0].plot(left_fitx, ploty,
                     color='#FF6B6B', linewidth=3, label='Left lane')

    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        axes[0].plot(right_fitx, ploty,
                     color='#4ECDC4', linewidth=3, label='Right lane')

    axes[0].axvline(x=w/2, color='yellow',
                    linewidth=2, linestyle='--', label='Vehicle position')

    if left_fit is not None and right_fit is not None:
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        axes[0].axvline(x=lane_center, color='lime',
                        linewidth=2, linestyle='--', label='Lane center')

    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Right plot: Statistics
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis('off')
    axes[1].set_title('Statistics', fontsize=12)
    axes[1].set_facecolor('#1a1a2e')

    info_lines = [
        ("Radius of Curvature:", "white"),
        (f"  Mean : {curvature_m:.1f} m" if curvature_m else "  Mean : N/A", "white"),
        ("", "white"),
        ("Road condition:", "white"),
        (f"  {get_curvature_description(curvature_m)}", "lime"),
        ("", "white"),
        ("Vehicle position:", "white"),
        (f"  {get_offset_description(offset_m)}", "orange"),
        ("", "white"),
        ("Conversion factors:", "white"),
        (f"  {YM_PER_PIX:.4f} m/px (vertical)", "white"),
        (f"  {XM_PER_PIX:.4f} m/px (horizontal)", "white"),
    ]

    y_pos = 9.5
    for line, color in info_lines:
        axes[1].text(0.5, y_pos, line, fontsize=13,
                     color=color, verticalalignment='top',
                     transform=axes[1].transData)
        y_pos -= 0.72

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved to: {save_path}")

    plt.show()


# =============================================================================
# MAIN - Run this module directly
# python src/curvature.py
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  CURVATURE MODULE - AdvancedLaneDetection")
    print("=" * 60)

    import sys
    sys.path.insert(0, '.')
    from src.calibration    import load_calibration, undistort_image
    from src.thresholding   import combined_threshold, region_of_interest
    from src.perspective    import (get_perspective_points,
                                    compute_perspective_transform,
                                    warp_image)
    from src.lane_detection import (find_lane_base, sliding_window_search,
                                    fit_polynomial)

    TEST_IMG = "data/test_images/test1.jpg"

    # Run previous pipeline steps
    print("\n[Running previous pipeline steps...]")
    mtx, dist = load_calibration('calibration_data.pkl')
    img       = cv2.imread(TEST_IMG)

    undistorted   = undistort_image(img, mtx, dist)
    binary        = combined_threshold(undistorted)
    binary_roi    = region_of_interest(binary)
    src, dst      = get_perspective_points(undistorted.shape)
    M, Minv       = compute_perspective_transform(src, dst)
    binary_warped = warp_image(binary_roi, M)

    leftx_base, rightx_base, _ = find_lane_base(binary_warped)
    leftx, lefty, rightx, righty, _ = sliding_window_search(
        binary_warped, leftx_base, rightx_base)
    left_fit, right_fit = fit_polynomial(
        leftx, lefty, rightx, righty, min_pixels=50)

    print("[OK] Previous steps completed!")

    # Calculate curvature
    print("\n[STEP 1] Calculating curvature...")
    left_curve, right_curve, mean_curve = calculate_curvature(
        left_fit, right_fit, binary_warped.shape)

    print(f"  Left  : {left_curve:.1f}m"  if left_curve  else "  Left  : Not detected")
    print(f"  Right : {right_curve:.1f}m" if right_curve else "  Right : Not detected")
    print(f"  Mean  : {mean_curve:.1f}m"  if mean_curve  else "  Mean  : N/A")
    if mean_curve:
        print(f"  -> {get_curvature_description(mean_curve)}")

    # Calculate offset
    print("\n[STEP 2] Calculating vehicle offset...")
    offset_m, offset_px = calculate_vehicle_offset(
        left_fit, right_fit, binary_warped.shape)

    if offset_m is not None:
        print(f"  Offset: {offset_m:+.3f}m ({offset_px:+.1f}px)")
        print(f"  -> {get_offset_description(offset_m)}")
    else:
        print("  Offset: N/A (need both lanes)")

    # Save visualization
    print("\n[STEP 3] Saving visualization...")
    os.makedirs("output/images", exist_ok=True)
    visualize_curvature(
        binary_warped.shape, left_fit, right_fit,
        mean_curve, offset_m,
        save_path="output/images/curvature_result.png"
    )
    print("\n[DONE] -> output/images/curvature_result.png")