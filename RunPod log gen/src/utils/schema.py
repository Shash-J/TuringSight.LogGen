def make_log(user_id, device_id, camera_id, event, object_text, timestamp):
    return {
        "user_id": user_id,
        "device_id": device_id,
        "camera_id": camera_id,
        "event": event,
        "object": object_text,
        "timestamp": int(timestamp)
    }