from datetime import timedelta, datetime


def pick_evenly_spaced_frames(frame_items, count):
    if not frame_items:
        return []

    if len(frame_items) <= count:
        return frame_items

    indices = []
    last_index = len(frame_items) - 1
    for i in range(count):
        idx = round(i * last_index / (count - 1))
        indices.append(idx)

    # remove duplicates if rounding caused repeats
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    return [frame_items[i] for i in unique_indices]


def build_activity_task(frame_item, prompt_version="activity_v1"):
    ts = frame_item["timestamp_dt"]
    ts_str = ts.isoformat()

    return {
        "task_id": f"activity_{int(ts.timestamp())}",
        "task_type": "activity_check",
        "priority": 3,
        "timestamp_utc": ts_str,
        "interval_start_utc": None,
        "interval_end_utc": None,
        "frame_paths": [frame_item["frame_path"]],
        "prompt_version": prompt_version
    }


def build_summary_task(task_type, end_dt, frame_items, frame_count, prompt_version, priority):
    selected = pick_evenly_spaced_frames(frame_items, frame_count)
    if not selected:
        return None

    start_dt = selected[0]["timestamp_dt"]

    return {
        "task_id": f"{task_type}_{int(end_dt.timestamp())}",
        "task_type": task_type,
        "priority": priority,
        "timestamp_utc": end_dt.isoformat(),
        "interval_start_utc": start_dt.isoformat(),
        "interval_end_utc": end_dt.isoformat(),
        "frame_paths": [x["frame_path"] for x in selected],
        "prompt_version": prompt_version
    }


def should_trigger(last_run_ts, now_ts, interval_sec):
    if last_run_ts is None:
        return True
    return (now_ts - last_run_ts) >= interval_sec


def build_event_task(event_data: dict, frame_buffer, frame_count: int = 4):
    """
    Builds an EventTask by coalescing multiple frames from the buffer
    leading up to the trigger event.
    """
    trigger_ts_float = event_data["trigger_ts"]
    end_dt = datetime.fromtimestamp(trigger_ts_float)
    # Grab frames from the last 15 seconds leading up to the trigger
    start_dt = end_dt - timedelta(seconds=15) 
    
    items = frame_buffer.get_between(start_dt, end_dt)
    selected = pick_evenly_spaced_frames(items, frame_count)
    
    if not selected:
        return None
        
    return {
        "task_id": f"event_{event_data['event_type']}_{event_data['entity_id']}_{int(trigger_ts_float)}",
        "task_type": "event_driven",
        "priority": 1, # High priority for active events
        "timestamp_utc": end_dt.isoformat(),
        "interval_start_utc": start_dt.isoformat(),
        "interval_end_utc": end_dt.isoformat(),
        "frame_paths": [x["frame_path"] for x in selected],
        "cv_event": event_data, # Pass structured CV memory to VLM
        "prompt_version": "event_dynamic"
    }