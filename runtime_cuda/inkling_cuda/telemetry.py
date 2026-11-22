"""Read-only NVIDIA telemetry checks for CUDA timing runs."""
import csv
import ctypes
import math
import os
import subprocess
from threading import Event, Thread
import time


FIELDS = {
    "power_w": "power.draw",
    "enforced_limit_w": "enforced.power.limit",
    "max_limit_w": "power.max_limit",
    "power_cap_us": "clocks_event_reasons_counters.sw_power_cap",
    "graphics_mhz": "clocks.gr",
    "memory_mhz": "clocks.mem",
    "gpu_utilization_percent": "utilization.gpu",
}


def uptime_seconds():
    if os.name == "nt":
        get_ticks = ctypes.windll.kernel32.GetTickCount64
        get_ticks.restype = ctypes.c_uint64
        return get_ticks() / 1000
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    return None


def parse_sample(output, uptime):
    row, = csv.reader(output.strip().splitlines(), skipinitialspace=True)
    sample = {}
    for key, raw in zip(FIELDS, row, strict=True):
        raw = raw.strip()
        if raw in ("N/A", "[N/A]", "[Not Supported]"):
            sample[key] = None
            continue
        value = int(raw) if key == "power_cap_us" else float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} reported an invalid value: {raw}")
        sample[key] = value

    issues = []
    power, max_limit = sample["power_w"], sample["max_limit_w"]
    if power is None:
        issues.append("GPU power draw is unavailable")
    # Allow margin over the board's reported maximum for transient spikes.
    elif power > (2 * max_limit if max_limit else 10_000):
        issues.append(f"Implausible GPU power draw: {power:g} W")
    duration = sample["power_cap_us"]
    if duration is not None and uptime is not None and duration > (uptime + 1) * 1_000_000:
        issues.append(f"GPU power-cap duration exceeds system uptime: {duration} us")
    utilization = sample["gpu_utilization_percent"]
    if utilization is not None and utilization > 100:
        issues.append(f"Invalid GPU utilization: {utilization:g}%")
    return sample, issues


def read_sample(device):
    command = ["nvidia-smi", "-i", str(device), "--query-gpu=" + ",".join(FIELDS.values()),
               "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=5)
        sample, issues = parse_sample(output, uptime_seconds())
    except subprocess.CalledProcessError as error:
        sample, issues = {}, [f"NVIDIA query failed ({error.returncode}): {error.output.strip()[:240]}"]
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        sample, issues = {}, [f"NVIDIA telemetry unavailable: {error}"]
    return {"timestamp": time.time(), **sample, "issues": issues}


class TelemetryWatch:
    def __init__(self, device):
        self.device = device
        self.samples = []
        self.stop = Event()
        self.worker = Thread(target=self.poll)

    def poll(self):
        while not self.stop.wait(1):
            self.samples.append(read_sample(self.device))

    def __enter__(self):
        sample = read_sample(self.device)
        self.samples.append(sample)
        if not sample["issues"]:
            self.worker.start()
        return self

    def __exit__(self, *_):
        if self.worker.ident is not None:
            self.stop.set()
            self.worker.join()
            self.samples.append(read_sample(self.device))


def report(device, samples):
    return {
        "device": str(device),
        "valid": all(not sample["issues"] for sample in samples),
        "samples": samples,
    }
