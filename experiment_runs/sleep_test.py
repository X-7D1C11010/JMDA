import pathlib
import time

pathlib.Path(r"D:\Code\JMDA-Net\experiment_runs\sleep_test_marker.txt").write_text(
    "started",
    encoding="utf-8",
)
print("sleep test started", flush=True)
time.sleep(30)
pathlib.Path(r"D:\Code\JMDA-Net\experiment_runs\sleep_test_done.txt").write_text(
    "done",
    encoding="utf-8",
)
