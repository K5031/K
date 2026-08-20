import threading

_lock = threading.Lock()
_initialized = False


def ensure_audio_ready():
    global _initialized
    with _lock:
        if not _initialized:
            import sounddevice as sd
            sd.query_devices() 
            _initialized = True


def resolve_output_device(preferred_names=("pipewire", "pulse", "default")):
    ensure_audio_ready()
    import sounddevice as sd
    devices = sd.query_devices()
    by_name = {
        d["name"]: i for i, d in enumerate(devices)
        if d.get("max_output_channels", 0) > 0
    }

    for name in preferred_names:
        if name in by_name:
            return by_name[name]

    default = sd.default.device
    if isinstance(default, (list, tuple)):
        default = default[1]
    if default is not None and default >= 0:
        return default

    raise RuntimeError("No usable audio output device found")


def resolve_pyaudio_input_device(preferred_names=("pipewire", "pulse", "default")):
    ensure_audio_ready()
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        by_name = {}
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                by_name[info["name"]] = i

        for name in preferred_names:
            if name in by_name:
                return by_name[name]

        return pa.get_default_input_device_info()["index"]
    finally:
        pa.terminate()