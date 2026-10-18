def run_detection(model, frames, conf_threshold=0.4, progress_callback=None):
    """
    Run YOLO prediction over extracted frame records.
    `model` can be a loaded YOLO model or a path to a .pt file.

    v2: each detection carries `segment_id` and `timestamp_sec` from its frame
    (GPS is no longer used), plus a running `det_id` used for labelling/evaluation.
    """
    if isinstance(model, str):
        from ultralytics import YOLO
        model = YOLO(model)

    raw_detections = []
    total_frames = len(frames)

    for i, rec in enumerate(frames):
        results = model.predict(rec["frame_path"], conf=conf_threshold, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_detections.append({
                "det_id": len(raw_detections),
                "frame": rec["frame_number"],
                "timestamp_sec": rec.get("timestamp_sec"),
                "segment_id": rec.get("segment_id", 1),
                "class": model.names[cls_id],
                "conf": round(float(box.conf[0]), 3),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                "frame_path": rec["frame_path"],
            })
        if progress_callback is not None and total_frames > 0:
            progress_callback(i + 1, total_frames)

    return raw_detections


if __name__ == "__main__":
    print("YOLO inference module loaded successfully.")
