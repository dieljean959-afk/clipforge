import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HOME", "/tmp/hfhome")

from faster_whisper import WhisperModel

try:
    m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=2)
    import numpy as np

    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(16000 * 10) * 0.001).astype(np.float32)
    segs, _ = m.transcribe(noise, language="es")
    for s in segs:
        pass
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            print("turbo peak RSS:", int(line.split()[1]) / 1024, "MB")
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:300])
