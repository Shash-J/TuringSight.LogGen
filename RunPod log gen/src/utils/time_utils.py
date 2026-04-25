from datetime import datetime, timezone


def now_utc():
    return datetime.now(timezone.utc)


def utc_iso(dt=None):
    if dt is None:
        dt = now_utc()
    return dt.isoformat()


def epoch_now():
    return int(now_utc().timestamp())


def safe_filename_timestamp(dt=None):
    if dt is None:
        dt = now_utc()
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def date_folder(dt=None):
    if dt is None:
        dt = now_utc()
    return dt.strftime("%Y-%m-%d")