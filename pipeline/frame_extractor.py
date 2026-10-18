import os
import cv2
import numpy as np

from pipeline.scene_segmenter import segment_for_frame, _thumb, DUPLICATE_DIFF

MAX_FRAME_INTERVAL = 15


def extract_frames(video_path, output_dir, frame_interval=5, cut_frames=None, skip_duplicates=True):
    """
    Extract every Nth frame from a video and save it to disk.

    v2 changes
    ----------
    * each record carries `segment_id` (from scene-cut detection) and `timestamp_sec`
    * `skip_duplicates`: frames identical to the previously saved frame are
      skipped. Frame-rate-converted videos repeat frames; counting the same
      picture twice would inflate an incident's evidence.
    * frame_interval is clamped to 1..MAX_FRAME_INTERVAL

    Returns (saved_frames, fps, stats)
    """
    frame_interval = int(max(1, min(frame_interval, MAX_FRAME_INTERVAL)))
    cut_frames = cut_frames or []
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    saved_frames = []
    duplicates_skipped = 0
    last_thumb = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            thumb = _thumb(frame)
            is_dup = (
                skip_duplicates
                and last_thumb is not None
                and float(np.mean(np.abs(thumb - last_thumb))) <= DUPLICATE_DIFF
            )
            if is_dup:
                duplicates_skipped += 1
            else:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append({
                    "frame_number": frame_count,
                    "timestamp_sec": round(frame_count / fps, 2),
                    "segment_id": segment_for_frame(frame_count, cut_frames),
                    "frame_path": frame_path,
                })
                last_thumb = thumb
        frame_count += 1

    cap.release()
    stats = {
        "total_video_frames": frame_count,
        "frame_interval": frame_interval,
        "frames_saved": len(saved_frames),
        "duplicates_skipped": duplicates_skipped,
    }
    return saved_frames, fps, stats


if __name__ == "__main__":
    print("Frame extractor module imported successfully.")
