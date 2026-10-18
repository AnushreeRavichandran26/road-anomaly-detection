def simulate_gps_from_start(frame_records, start_lat, start_lon, assumed_speed_kmh, fps, frame_interval):
    """
    Only a starting GPS point is available (no real GPS log / metadata track),
    so position is estimated frame-by-frame assuming constant speed and a
    fixed heading (north).
    """
    speed_m_s = assumed_speed_kmh * 1000 / 3600
    seconds_per_extracted_frame = frame_interval / fps if fps > 0 else 0
    meters_per_frame_step = speed_m_s * seconds_per_extracted_frame
    meters_per_degree_lat = 111320.0

    for idx, rec in enumerate(frame_records):
        delta_lat = (meters_per_frame_step * idx) / meters_per_degree_lat
        rec["gps"] = (round(start_lat + delta_lat, 6), round(start_lon, 6))
    return frame_records


if __name__ == "__main__":
    test_records = [{"frame_number": 0}, {"frame_number": 5}]
    res = simulate_gps_from_start(test_records, 13.0827, 80.2707, 30, 25.0, 5)
    print("GPS tagger smoke test result:", res)
