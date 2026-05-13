# =============================================================================
# Module: pipeline.py
# Mục đích: Ghép tất cả modules lại thành pipeline hoàn chỉnh
#
# Đây là module trung tâm của project:
# - Khởi tạo tất cả components (calibration, transform...)
# - Xử lý từng frame video qua toàn bộ pipeline
# - Warp kết quả ngược lại lên ảnh gốc
# - Tô màu vùng làn đường
# - Hiển thị thông số curvature và offset
# - Xuất video kết quả
#
# Pipeline flow:
#   Frame → Undistort → Threshold → Warp → Detect → Curvature
#         → Unwarp → Draw → Output
# =============================================================================

import cv2
import numpy as np
import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration    import load_calibration, undistort_image
from src.thresholding   import combined_threshold, region_of_interest
from src.perspective    import (get_perspective_points,
                                compute_perspective_transform,
                                warp_image, unwarp_image)
from src.lane_detection import (LaneTracker, detect_lanes,
                                draw_lane_area)
from src.curvature      import (calculate_curvature,
                                calculate_vehicle_offset,
                                draw_info_on_frame)


# =============================================================================
# CLASS: LaneDetectionPipeline
# Mục đích: Đóng gói toàn bộ pipeline vào một class dễ dùng
# =============================================================================
class LaneDetectionPipeline:
    """
    Pipeline hoàn chỉnh cho Advanced Lane Detection.

    Cách dùng:
        pipeline = LaneDetectionPipeline()
        pipeline.initialize()
        result = pipeline.process_frame(frame)
    """

    def __init__(self,
                 calibration_path='calibration_data.pkl',
                 n_frames=5,
                 ema_alpha=0.25):
        """
        Tham số:
            calibration_path : Đường dẫn file calibration đã lưu
            n_frames         : Số frame để tính trung bình (smoothing)
            ema_alpha        : Hệ số EMA cho smoothing
        """
        self.calibration_path = calibration_path
        self.n_frames         = n_frames
        self.ema_alpha        = ema_alpha

        # Các thành phần sẽ được khởi tạo trong initialize()
        self.mtx   = None   # Camera matrix
        self.dist  = None   # Distortion coefficients
        self.M     = None   # Perspective transform matrix
        self.Minv  = None   # Inverse perspective transform
        self.src   = None   # Source points
        self.dst   = None   # Destination points

        # Lane trackers — lưu trạng thái qua các frame
        self.left_tracker  = None
        self.right_tracker = None

        # Thống kê
        self.frame_count    = 0
        self.success_count  = 0
        self.total_time     = 0.0

        self.initialized = False

    # ------------------------------------------------------------------
    def initialize(self, img_shape=None,
                   src_points=None, dst_points=None):
        """
        Khởi tạo pipeline — gọi một lần trước khi xử lý video.

        Tham số:
            img_shape  : (height, width) của frame video
                         Nếu None → đọc từ frame đầu tiên
            src_points : Điểm SRC tùy chỉnh (None = tự tính)
            dst_points : Điểm DST tùy chỉnh (None = tự tính)
        """
        print("[PIPELINE] Khởi tạo...")

        # BƯỚC 1: Load camera calibration
        print("  [1/4] Load camera calibration...")
        self.mtx, self.dist = load_calibration(self.calibration_path)
        if self.mtx is None:
            raise RuntimeError(
                f"Không load được calibration từ: {self.calibration_path}\n"
                "Hãy chạy src/calibration.py trước!"
            )

        # BƯỚC 2: Tính perspective transform
        if img_shape is not None:
            print("  [2/4] Tính perspective transform...")
            self.src, self.dst = get_perspective_points(
                img_shape, src_points, dst_points
            )
            self.M, self.Minv = compute_perspective_transform(
                self.src, self.dst
            )

        # BƯỚC 3: Khởi tạo Lane Trackers
        print("  [3/4] Khởi tạo Lane Trackers...")
        self.left_tracker  = LaneTracker(
            n_frames=self.n_frames,
            ema_alpha=self.ema_alpha
        )
        self.right_tracker = LaneTracker(
            n_frames=self.n_frames,
            ema_alpha=self.ema_alpha
        )

        # BƯỚC 4: Reset thống kê
        print("  [4/4] Reset thống kê...")
        self.frame_count   = 0
        self.success_count = 0
        self.total_time    = 0.0

        self.initialized = True
        print("[PIPELINE] Khởi tạo hoàn thành!\n")

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Xử lý một frame video qua toàn bộ pipeline.

        Tham số:
            frame : Ảnh BGR từ cv2.VideoCapture

        Trả về:
            result      : Frame kết quả với làn đường và thông số
            debug_info  : Dictionary chứa thông tin debug
        """
        t_start = time.time()
        self.frame_count += 1

        # Khởi tạo perspective transform nếu chưa có
        if self.M is None:
            self.src, self.dst = get_perspective_points(frame.shape)
            self.M, self.Minv  = compute_perspective_transform(
                self.src, self.dst
            )

        debug_info = {
            'frame_id'    : self.frame_count,
            'left_fit'    : None,
            'right_fit'   : None,
            'curvature_m' : None,
            'offset_m'    : None,
            'success'     : False,
        }

        try:
            # ==============================================================
            # BƯỚC 1: Undistort
            # Loại bỏ méo hình từ camera lens
            # ==============================================================
            undistorted = undistort_image(frame, self.mtx, self.dist)

            # ==============================================================
            # BƯỚC 2: Threshold
            # Tạo binary image highlight làn đường
            # ==============================================================
            binary     = combined_threshold(undistorted)
            binary_roi = region_of_interest(binary)

            # ==============================================================
            # BƯỚC 3: Perspective Transform
            # Chuyển sang Birds-Eye View
            # ==============================================================
            binary_warped = warp_image(binary_roi, self.M)

            # ==============================================================
            # BƯỚC 4: Lane Detection
            # Tìm lane pixels + fit polynomial + smoothing
            # ==============================================================
            left_fit, right_fit, _, method = detect_lanes(
                binary_warped,
                self.left_tracker,
                self.right_tracker
            )

            debug_info['left_fit']  = left_fit
            debug_info['right_fit'] = right_fit
            debug_info['method']    = method

            # ==============================================================
            # BƯỚC 5: Tính Curvature và Offset
            # ==============================================================
            left_curve, right_curve, mean_curve = calculate_curvature(
                left_fit, right_fit, binary_warped.shape
            )
            offset_m, offset_px = calculate_vehicle_offset(
                left_fit, right_fit, binary_warped.shape
            )

            debug_info['curvature_m']   = mean_curve
            debug_info['offset_m']      = offset_m
            debug_info['left_curverad'] = left_curve
            debug_info['right_curverad']= right_curve

            # ==============================================================
            # BƯỚC 6: Vẽ kết quả lên ảnh
            # ==============================================================
            # 6a. Tô màu vùng làn đường trên Birds-Eye View
            lane_warped = draw_lane_area(binary_warped, left_fit, right_fit)

            # 6b. Warp ngược về góc nhìn camera gốc
            lane_unwarped = unwarp_image(
                lane_warped, self.Minv, undistorted.shape
            )

            # 6c. Ghép vùng làn đường lên ảnh gốc
            result = cv2.addWeighted(undistorted, 1.0,
                                     lane_unwarped, 0.3, 0)

            # 6d. Vẽ thông số curvature và offset lên frame
            result = draw_info_on_frame(
                result, mean_curve, offset_m,
                left_curve, right_curve
            )

            # 6e. Vẽ thêm thông tin debug nhỏ ở góc phải
            result = self._draw_debug_info(result, method, binary_warped)

            debug_info['success'] = True
            self.success_count   += 1

        except Exception as e:
            # Nếu có lỗi → trả về frame gốc với thông báo lỗi
            print(f"  [LỖI] Frame {self.frame_count}: {e}")
            result = frame.copy()
            cv2.putText(result, f"LOI: {str(e)[:50]}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # Tính thời gian xử lý
        t_end = time.time()
        proc_time = t_end - t_start
        self.total_time += proc_time
        debug_info['proc_time_ms'] = proc_time * 1000

        return result, debug_info

    # ------------------------------------------------------------------
    def _draw_debug_info(self, frame, method, binary_warped):
        """
        Vẽ thông tin debug nhỏ ở góc phải trên (thumbnail + method).

        Tham số:
            frame         : Frame để vẽ lên
            method        : Phương pháp detect ("Sliding Window" / "Around Poly")
            binary_warped : Binary warped để hiển thị thumbnail
        """
        h, w = frame.shape[:2]

        # Thumbnail binary warped ở góc phải trên
        thumb_w, thumb_h = 200, 120
        thumb = cv2.resize(binary_warped, (thumb_w, thumb_h))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

        # Vị trí thumbnail
        x_offset = w - thumb_w - 10
        y_offset = 10
        frame[y_offset:y_offset+thumb_h,
              x_offset:x_offset+thumb_w] = thumb_bgr

        # Viền thumbnail
        cv2.rectangle(frame,
                      (x_offset, y_offset),
                      (x_offset+thumb_w, y_offset+thumb_h),
                      (255, 255, 255), 1)

        # Label thumbnail
        cv2.putText(frame, "Binary Warped",
                    (x_offset + 5, y_offset + thumb_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)

        # Phương pháp detect
        method_color = (0, 255, 0) if "Poly" in method else (0, 200, 255)
        cv2.putText(frame, f"Method: {method}",
                    (x_offset, y_offset + thumb_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    method_color, 1)

        # Frame counter
        cv2.putText(frame,
                    f"Frame: {self.frame_count} | "
                    f"OK: {self.success_count}",
                    (x_offset, y_offset + thumb_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1)

        return frame

    # ------------------------------------------------------------------
    def get_statistics(self):
        """
        Trả về thống kê xử lý video.
        """
        if self.frame_count == 0:
            return {}

        avg_time = self.total_time / self.frame_count * 1000
        success_rate = self.success_count / self.frame_count * 100

        return {
            'total_frames'  : self.frame_count,
            'success_frames': self.success_count,
            'success_rate'  : success_rate,
            'avg_time_ms'   : avg_time,
            'fps_estimate'  : 1000 / avg_time if avg_time > 0 else 0,
        }

    # ------------------------------------------------------------------
    def reset(self):
        """Reset trackers để bắt đầu video mới."""
        if self.left_tracker:
            self.left_tracker.reset()
        if self.right_tracker:
            self.right_tracker.reset()
        self.frame_count   = 0
        self.success_count = 0
        self.total_time    = 0.0
        print("[PIPELINE] Đã reset.")


# =============================================================================
# HÀM: process_video()
# Mục đích: Xử lý toàn bộ video file
# =============================================================================
def process_video(input_path, output_path, pipeline,
                  show_preview=True, max_frames=None):
    """
    Xử lý video qua pipeline và lưu kết quả.

    Tham số:
        input_path  : Đường dẫn video đầu vào
        output_path : Đường dẫn video kết quả
        pipeline    : LaneDetectionPipeline đã initialize()
        show_preview: True để hiển thị preview real-time
        max_frames  : Số frame tối đa (None = xử lý hết)
    """
    # Mở video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_path}")

    # Thông số video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VIDEO] Input : {input_path}")
    print(f"  Kích thước  : {width}x{height}")
    print(f"  FPS         : {fps:.1f}")
    print(f"  Tổng frames : {total_frames}")

    # Tạo VideoWriter để lưu output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Khởi tạo pipeline với kích thước frame
    pipeline.initialize(img_shape=(height, width))

    print(f"\n[VIDEO] Bắt đầu xử lý...")
    print("  Nhấn 'q' để dừng | 'p' để pause | 's' để screenshot")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Kiểm tra giới hạn frames
            if max_frames and pipeline.frame_count >= max_frames:
                break

        # Xử lý frame qua pipeline
        result, debug_info = pipeline.process_frame(frame)

        # Ghi frame kết quả vào video output
        out.write(result)

        # Hiển thị preview
        if show_preview:
            # Scale xuống để hiển thị nếu quá lớn
            preview = result
            if width > 1280:
                scale   = 1280 / width
                preview = cv2.resize(result,
                                     (int(width*scale), int(height*scale)))

            cv2.imshow('Advanced Lane Detection', preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Dừng theo yêu cầu người dùng.")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"  [{status}]")
            elif key == ord('s'):
                # Screenshot
                ss_path = f"output/images/screenshot_frame{pipeline.frame_count}.png"
                cv2.imwrite(ss_path, result)
                print(f"  [Screenshot] Lưu vào: {ss_path}")

        # In tiến độ mỗi 50 frames
        if pipeline.frame_count % 50 == 0:
            stats = pipeline.get_statistics()
            print(f"  Frame {pipeline.frame_count}/{total_frames} | "
                  f"OK: {stats['success_rate']:.1f}% | "
                  f"~{stats['fps_estimate']:.1f} FPS")

    # Dọn dẹp
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    # In thống kê cuối
    stats = pipeline.get_statistics()
    print(f"\n[KẾT QUẢ]")
    print(f"  Tổng frames xử lý : {stats['total_frames']}")
    print(f"  Frames thành công  : {stats['success_frames']}")
    print(f"  Tỷ lệ thành công   : {stats['success_rate']:.1f}%")
    print(f"  Thời gian TB/frame : {stats['avg_time_ms']:.1f} ms")
    print(f"  FPS ước tính       : {stats['fps_estimate']:.1f}")
    print(f"  Video lưu tại      : {output_path}")

    return stats


# =============================================================================
# HÀM: process_image()
# Mục đích: Xử lý một ảnh tĩnh qua pipeline (để test)
# =============================================================================
def process_image(img_path, output_path, pipeline):
    """
    Xử lý một ảnh tĩnh qua pipeline.

    Tham số:
        img_path    : Đường dẫn ảnh đầu vào
        output_path : Đường dẫn ảnh kết quả
        pipeline    : LaneDetectionPipeline
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[LỖI] Không đọc được: {img_path}")
        return None

    # Khởi tạo pipeline
    pipeline.initialize(img_shape=img.shape)

    # Xử lý
    result, debug_info = pipeline.process_frame(img)

    # Lưu kết quả
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)

    print(f"[OK] Ảnh kết quả lưu tại: {output_path}")
    print(f"  Curvature : {debug_info.get('curvature_m', 'N/A')}")
    print(f"  Offset    : {debug_info.get('offset_m', 'N/A')}")

    return result, debug_info


# =============================================================================
# CHẠY THỬ MODULE NÀY TRỰC TIẾP
# Khi chạy: python src/pipeline.py
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  PIPELINE MODULE - AdvancedLaneDetection")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Cấu hình
    # ------------------------------------------------------------------
    TEST_IMG   = "data/test_images/test1.jpg"
    TEST_VIDEO = "data/test_videos/project_video.mp4"
    OUT_IMG    = "output/images/pipeline_result.png"
    OUT_VIDEO  = "output/videos/result.mp4"

    # Tạo pipeline
    pipeline = LaneDetectionPipeline(
        calibration_path='calibration_data.pkl',
        n_frames=5,
        ema_alpha=0.25
    )

    # ------------------------------------------------------------------
    # TEST 1: Xử lý ảnh tĩnh
    # ------------------------------------------------------------------
    print("\n[TEST 1] Xử lý ảnh tĩnh...")
    if os.path.exists(TEST_IMG):
        result, info = process_image(TEST_IMG, OUT_IMG, pipeline)

        if result is not None:
            cv2.imshow("Pipeline Result - Image", result)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
    else:
        print(f"  [SKIP] Không tìm thấy: {TEST_IMG}")

    # ------------------------------------------------------------------
    # TEST 2: Xử lý video
    # ------------------------------------------------------------------
    print("\n[TEST 2] Xử lý video...")
    if os.path.exists(TEST_VIDEO):
        # Reset pipeline trước khi xử lý video
        pipeline.reset()

        stats = process_video(
            input_path   = TEST_VIDEO,
            output_path  = OUT_VIDEO,
            pipeline     = pipeline,
            show_preview = True,
            max_frames   = None   # None = xử lý hết video
        )
    else:
        print(f"  [SKIP] Không tìm thấy: {TEST_VIDEO}")
        print(f"  Đặt video vào: {TEST_VIDEO}")

    print("\n[HOÀN THÀNH] Pipeline sẵn sàng!")


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
#                                 │ warped binary + M, Minv
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 4: lane_detection.py ✅ XONG                                 │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ left_fit, right_fit
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 5: curvature.py     ✅ XONG                                  │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │ curvature_m, offset_m
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 6: pipeline.py      ← BẠN ĐANG Ở ĐÂY                       │
#  │  - LaneDetectionPipeline  → class đóng gói toàn bộ pipeline       │
#  │  - process_frame()        → xử lý 1 frame qua toàn bộ bước        │
#  │  - unwarp + draw          → chiếu kết quả lên ảnh gốc             │
#  │  - draw_info_on_frame()   → curvature + offset lên video           │
#  │  - process_video()        → xử lý toàn bộ video file              │
#  │  - process_image()        → test trên ảnh tĩnh                    │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │
#                                 ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  BƯỚC 7: main.py           (cuối cùng)                             │
#  │  Entry point — chạy toàn bộ project từ 1 lệnh                     │
#  └──────────────────────────────┬──────────────────────────────────────┘
#                                 │
#                                 ▼
#                        [VIDEO KẾT QUẢ]
#
# =============================================================================