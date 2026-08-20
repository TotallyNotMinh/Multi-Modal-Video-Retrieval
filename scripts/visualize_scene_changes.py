import os
import sys
import time
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Ensure repo root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoding.scene_detector import SceneDetector



def draw_hud(
    frame: np.ndarray,
    curr_frame: int,
    total_frames: int,
    fps: float,
    current_shot: dict,
    delta: float,
    threshold: float,
    is_cut: bool,
    alert_countdown: int,
    cut_info: dict,
    is_sampled_keyframe: bool
) -> np.ndarray:
    """
    Renders a semi-transparent HUD and visual alert overlays onto the video frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top HUD semi-transparent background (height: 70px)
    hud_h = 70
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 2. Text styling
    pts_sec = curr_frame / fps
    total_sec = total_frames / fps
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Row 1: Timing and Shot Tracker
    shot_id = current_shot["shot_id"] if current_shot else 0
    shot_dur = current_shot["duration_sec"] if current_shot else 0.0
    text_row1 = f"Frame: {curr_frame:05d}/{total_frames:05d} ({pts_sec:06.2f}s/{total_sec:06.2f}s)  |  Current Shot: #{shot_id:03d} (Duration: {shot_dur:04.1f}s)"
    cv2.putText(frame, text_row1, (16, 26), font, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    # Row 2: Live Histogram Delta & Metric
    text_row2 = f"Delta: {delta:05.3f} (Threshold: {threshold:04.2f})"
    cv2.putText(frame, text_row2, (16, 54), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Row 2 Meter / Gauge
    meter_x = 360
    meter_y = 42
    meter_w = 200
    meter_h = 14
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 80), 1)

    # Gauge fill level based on delta/threshold
    fill_ratio = min(1.0, max(0.0, delta / max(0.01, threshold * 1.5)))
    fill_w = int(fill_ratio * meter_w)
    
    if delta >= threshold:
        gauge_color = (0, 0, 255)      # Red (Cut trigger)
    elif delta >= threshold * 0.7:
        gauge_color = (0, 215, 255)    # Yellow/Orange (Close to cut)
    else:
        gauge_color = (0, 220, 0)      # Green (Stable scene)

    if fill_w > 0:
        cv2.rectangle(frame, (meter_x + 1, meter_y + 1), (meter_x + fill_w, meter_y + meter_h - 1), gauge_color, -1)

    # Threshold marker line on gauge
    thresh_x = meter_x + int((1.0 / 1.5) * meter_w)
    cv2.line(frame, (thresh_x, meter_y - 2), (thresh_x, meter_y + meter_h + 2), (255, 255, 255), 2)

    # 3. Active Cut Alert Banner & Frame Border Flash
    if alert_countdown > 0:
        # Red Border
        border_thickness = 4
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)

        # Cut Banner Box across center-top
        banner_w = 520
        banner_h = 42
        bx1 = (w - banner_w) // 2
        by1 = hud_h + 12
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        # Semi-transparent red banner background
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

        cut_id = cut_info.get("shot_id", shot_id)
        cut_delta = cut_info.get("delta", delta)
        banner_text = f"!! SCENE CHANGE DETECTED -> Shot #{cut_id} (Delta: {cut_delta:.3f})"
        cv2.putText(frame, banner_text, (bx1 + 16, by1 + 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Sampled Keyframe Badge (Right-aligned badge)
    if is_sampled_keyframe:
        badge_w = 260
        badge_h = 32
        kx1 = w - badge_w - 16
        ky1 = hud_h + 12
        kx2 = kx1 + badge_w
        ky2 = ky1 + badge_h

        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (kx1, ky1), (kx2, ky2), (0, 140, 255), -1)  # Gold/Orange
        cv2.addWeighted(badge_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (255, 255, 255), 1)

        keyframe_text = "* SAMPLED KEYFRAME"
        cv2.putText(frame, keyframe_text, (kx1 + 14, ky1 + 22), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def visualize_scene_changes(
    video_path: str,
    output_path: str = "debug_outputs/annotated_scene_cuts.mp4",
    threshold: float = 0.35,
    min_shot_frames: int = 3,
    max_seconds: float = None,
    alert_duration: int = 8,
):
    """
    Renders the video with real-time scene change detection annotations and keyframe indicators.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[Visualizer] Initializing SceneDetector (threshold={threshold}, min_frames={min_shot_frames})...")
    detector = SceneDetector(threshold=threshold, min_shot_frames=min_shot_frames)

    # Pass 1: Run Scene Detection
    t0 = time.time()
    shots = detector.detect_shots(video_path)
    elapsed_detect = time.time() - t0
    print(f"[Visualizer] Detected {len(shots)} shots in {elapsed_detect:.2f}s.")

    # Open video to get metadata
    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if (raw_fps and not np.isnan(raw_fps) and raw_fps > 0) else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Calculate frame limits
    max_frames = int(max_seconds * fps) if max_seconds else total_frames
    frames_to_process = min(total_frames, max_frames)

    # Build lookup tables
    cut_map = {}
    for shot in shots:
        if shot["start_frame"] > 0:
            cut_map[shot["start_frame"]] = shot

    sampled_keyframes = set()
    for shot in shots:
        for f in detector.get_sample_frames(shot, fps):
            sampled_keyframes.add(f)

    print(f"[Visualizer] Video: {width}x{height} @ {fps:.1f} fps ({total_frames} total frames).")
    print(f"[Visualizer] Rendering up to {frames_to_process} frames ({frames_to_process/fps:.1f}s) to {output_path}...")

    # Configure VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_hist = None
    curr_shot_idx = 0
    alert_countdown = 0
    last_cut_info = {}

    pbar = tqdm(total=frames_to_process, desc="Rendering annotated video")

    curr_frame = 0
    try:
        while curr_frame < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute histogram delta exactly matching SceneDetector logic
            small = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            delta = 0.0
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                delta = max(0.0, 1.0 - corr)
            prev_hist = hist

            # Update current shot tracker
            while curr_shot_idx + 1 < len(shots) and curr_frame >= shots[curr_shot_idx + 1]["start_frame"]:
                curr_shot_idx += 1
            current_shot = shots[curr_shot_idx] if shots else None

            # Check if current frame triggers a scene cut
            is_cut = curr_frame in cut_map
            if is_cut:
                alert_countdown = alert_duration
                last_cut_info = {"shot_id": cut_map[curr_frame]["shot_id"], "delta": delta}

            is_sampled = curr_frame in sampled_keyframes

            # Render HUD & Overlays
            annotated_frame = draw_hud(
                frame=frame,
                curr_frame=curr_frame,
                total_frames=total_frames,
                fps=fps,
                current_shot=current_shot,
                delta=delta,
                threshold=threshold,
                is_cut=is_cut,
                alert_countdown=alert_countdown,
                cut_info=last_cut_info,
                is_sampled_keyframe=is_sampled,
            )

            if alert_countdown > 0:
                alert_countdown -= 1

            writer.write(annotated_frame)
            pbar.update(1)
            curr_frame += 1
    finally:
        pbar.close()
        cap.release()
        writer.release()


    print(f"\n[Visualizer] Finished! Annotated video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video with real-time Scene Cut HUD and Keyframe markers.")
    parser.add_argument("--video-path", type=str, default="data/Videos_L21_a/video/L21_V001.mp4",
                        help="Path to input .mp4 video.")
    parser.add_argument("--output-path", type=str, default="debug_outputs/L21_V001_scene_cuts.mp4",
                        help="Path to output annotated .mp4 video.")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Histogram correlation drop threshold (0.0 - 1.0).")
    parser.add_argument("--min-shot-frames", type=int, default=3,
                        help="Minimum frames per shot.")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Optional maximum seconds of video to render (e.g. 30 for quick preview).")
    parser.add_argument("--alert-duration", type=int, default=8,
                        help="Number of frames to display the cut alert banner.")
    args = parser.parse_args()

    visualize_scene_changes(
        video_path=args.video_path,
        output_path=args.output_path,
        threshold=args.threshold,
        min_shot_frames=args.min_shot_frames,
        max_seconds=args.max_seconds,
        alert_duration=args.alert_duration,
    )
