import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inkling_cuda.__main__ import generate
from inkling_cuda.telemetry import parse_sample, read_sample


class TelemetryTests(unittest.TestCase):
    def test_idle_clock_and_unavailable_optional_fields_are_valid(self):
        sample, issues = parse_sample("25.35, [N/A], 150, 404062, 180, 405, 0", 10_000)
        self.assertEqual(issues, [])
        self.assertIsNone(sample["enforced_limit_w"])
        self.assertEqual(sample["graphics_mhz"], 180)

    def test_reports_both_implausible_readings(self):
        sample, issues = parse_sample("1059127, 149.99, 150, 1795329866932383, 180, 14001, 0", 10_000)
        self.assertEqual(sample["power_w"], 1059127)
        self.assertEqual(len(issues), 2)
        self.assertIn("Implausible GPU power draw", issues[0])
        self.assertIn("exceeds system uptime", issues[1])

    def test_query_error_produces_an_invalid_sample(self):
        error = subprocess.CalledProcessError(1, "nvidia-smi", output="Driver query failed\n")
        with patch("inkling_cuda.telemetry.subprocess.check_output", side_effect=error):
            sample = read_sample("0000:c5:00.0")
        self.assertIn("Driver query failed", sample["issues"][0])

    def test_invalid_preflight_skips_generation(self):
        result, run = self.checked_run([{"issues": ["Implausible GPU power draw"]}])
        run.assert_not_called()
        self.assertIsNone(result["decode_tokens_per_second"])

    def test_invalid_final_read_retains_tokens_and_withholds_decode_score(self):
        result, run = self.checked_run([{"issues": []}, {"issues": ["GPU power draw is unavailable"]}])
        run.assert_called_once()
        self.assertEqual(result["generated_tokens"], [17])
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertFalse(result["telemetry"]["valid"])

    def checked_run(self, samples):
        args = SimpleNamespace(model=Path("unused"), tokens=[4], max_new_tokens=2, context=2, device=0,
                               check_telemetry=True, json=None)
        with (
            patch("inkling_cuda.__main__.Cuda") as cuda,
            patch("inkling_cuda.__main__.generate_result",
                  return_value={"generated_tokens": [17], "decode_tokens_per_second": 15.1}) as run,
            patch("inkling_cuda.telemetry.read_sample", side_effect=samples),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            self.assertRaises(SystemExit) as raised,
        ):
            device = cuda.return_value.__enter__.return_value
            device.pci_bus_id.return_value = "0000:c5:00.0"
            device.device_name.return_value = "NVIDIA GPU"
            generate(args)
        self.assertEqual(raised.exception.code, 2)
        return json.loads(stdout.getvalue()), run
