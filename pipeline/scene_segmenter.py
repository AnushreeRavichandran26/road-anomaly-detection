"""
Scene-cut detection (v2).

Why: a dashcam upload may be several clips joined together (different roads).
Detections from two different roads must never be merged into one incident,
so every frame gets a `segment_id` and the Verification Agent resets its
grouping at every segment boundary.

How: each frame is shrunk to a 96x72 grayscale thumbnail and compared with the
previous *distinct* frame (mean absolute pixel difference). Normal driving
produces a steady level of change; a hard cut produces a spike far above the
local median. Exact duplicate frames (common after frame-rate conversion) are
ignored so they don't distort the median.
"""
import cv2
import numpy as np

THUMB_SIZE = (96, 72)
DUPLICATE_DIFF = 1.0     # mean abs diff below this = duplicate/frozen frame


def _thumb(frame):
    small = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def detect_scene_cuts(video_path, sensitivity=4.0, window=10, min_scene_sec=1.0):
    """
    Return (cut_frames, fps, total_frames).

    cut_frames  : sorted list of frame numbers where a new segment starts
                  (frame 0 is implicit and not included).
    sensitivity : a frame is a cut if its change is `sensitivity` times the
                  local median change. Lower = more cuts. 4.0 ignores normal
                  motion and exposure changes but catches hard cuts.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    diffs = []          # (frame_number, diff) for distinct frames only
    prev = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = _thumb(frame)
        if prev is not None:
            d = float(np.mean(np.abs(t - prev)))
            if d > DUPLICATE_DIFF:
                diffs.append((n, d))
        prev = t
        n += 1
    cap.release()

    if len(diffs) < 3:
        return [], fps, n

    values = np.array([d for _, d in diffs])
    min_gap = int(min_scene_sec * fps)
    cuts = []
    for j, (frame_no, d) in enumerate(diffs):
        neighbours = np.r_[values[max(0, j - window):j], values[j + 1:j + 1 + window]]
        local_median = float(np.median(neighbours)) if len(neighbours) else 0.0
        if local_median > 0 and d / local_median >= sensitivity:
            if not cuts or frame_no - cuts[-1] >= min_gap:
                cuts.append(frame_no)
    return cuts, fps, n


def segment_for_frame(frame_number, cut_frames):
    """Segment ids start at 1. Segment k starts at cut_frames[k-2]."""
    seg = 1
    for c in cut_frames:
        if frame_number >= c:
            seg += 1
        else:
            break
    return seg


def build_segments(cut_frames, fps, total_frames):
    """List of {segment_id, start_frame, end_frame, start_sec, end_sec}."""
    bounds = [0] + list(cut_frames) + [total_frames]
    segs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1] - 1
        segs.append({
            "segment_id": i + 1,
            "start_frame": s,
            "end_frame": e,
            "start_sec": round(s / fps, 2),
            "end_sec": round(e / fps, 2),
        })
    return segs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cuts, fps, total = detect_scene_cuts(sys.argv[1])
        print(f"fps={fps} frames={total} cuts={cuts}")
        for s in build_segments(cuts, fps, total):
            print(s)
    else:
        print("Usage: python -m pipeline.scene_segmenter <video>")
