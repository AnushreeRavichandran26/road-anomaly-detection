"""A stand-in for the YOLO model so the pipeline can be tested without ultralytics.
It 'detects' defects on a fixed schedule of frames, including a one-frame false
positive, a same-frame second class, and a repaired patch."""
import os, re

class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]; self.conf = [conf]
        self.xyxy = [type("T", (), {"tolist": lambda s, v=xyxy: list(v)})()]

class _Result:
    def __init__(self, boxes): self.boxes = boxes

class FakeYOLO:
    names = {0: "D00_longitudinal_crack", 1: "D10_transverse_crack", 2: "D20_alligator_crack",
             3: "D40_pothole", 4: "repair"}
    # frame ranges (raw frame numbers) where each class is "visible"
    schedule = [
        (0, 60, 90, 0.55),     # crack, seen for 1 s
        (3, 75, 95, 0.80),     # pothole overlapping the crack (same frames)
        (1, 300, 300, 0.45),   # single-frame false positive -> must be rejected
        (4, 450, 500, 0.70),   # repaired patch -> ignored
        (3, 600, 640, 0.35),   # low-confidence pothole (filtered by 0.4, kept by 0.25)
        (2, 800, 820, 0.65),   # alligator crack
    ]
    def predict(self, path, conf=0.25, verbose=False):
        f = int(re.search(r"frame_(\d+)", os.path.basename(path)).group(1))
        boxes = [_Box(c, p, (100, 500, 300, 600)) for c, a, b, p in self.schedule if a <= f <= b and p >= conf]
        return [_Result(boxes)]
