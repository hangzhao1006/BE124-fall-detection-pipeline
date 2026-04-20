"""
BE124 Ankle Exoskeleton - Single ESP32 BLE Data Logger
======================================================
Connects to one ESP32 (BE124_IMU) via BLE.
Receives data from 3 IMU channels (THIGH, SHANK, FOOT)
via 3 separate BLE characteristics.

Usage:
  python ble_logger.py --trial slip_hang_01
  python ble_logger.py --test

Controls:
  SPACE = mark perturbation event
  Q     = stop and save

Requirements:
  pip install bleak
"""

import asyncio
import time
import csv
import os
import sys
import argparse
from datetime import datetime
from bleak import BleakScanner, BleakClient

# ============================================================
# BLE Config (must match firmware)
# ============================================================
DEVICE_NAME = "BE124_IMU"
SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
CHAR_THIGH   = "abcd0001-ab12-cd34-ef56-abcdef123456"
CHAR_SHANK   = "abcd0002-ab12-cd34-ef56-abcdef123456"
CHAR_FOOT    = "abcd0003-ab12-cd34-ef56-abcdef123456"

OUTPUT_DIR = "data"

CSV_COLUMNS = [
    "timestamp",
    "thigh_acc_x", "thigh_acc_y", "thigh_acc_z",
    "thigh_gyro_x", "thigh_gyro_y", "thigh_gyro_z",
    "thigh_mag_x", "thigh_mag_y", "thigh_mag_z",
    "shank_acc_x", "shank_acc_y", "shank_acc_z",
    "shank_gyro_x", "shank_gyro_y", "shank_gyro_z",
    "shank_mag_x", "shank_mag_y", "shank_mag_z",
    "foot_acc_x", "foot_acc_y", "foot_acc_z",
    "foot_gyro_x", "foot_gyro_y", "foot_gyro_z",
    "foot_mag_x", "foot_mag_y", "foot_mag_z",
    "foot_euler_x", "foot_euler_y", "foot_euler_z",
    "perturbation_event",
]


class IMULogger:
    def __init__(self, trial_name=None):
        self.latest = {"thigh": None, "shank": None, "foot": None}
        self.counts = {"thigh": 0, "shank": 0, "foot": 0}
        self.buffer = []
        self.perturb_flag = False
        self.running = True
        self.client = None

        if trial_name is None:
            trial_name = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trial_name = trial_name
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _make_handler(self, key):
        def handler(sender, data):
            try:
                line = data.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    return
                parts = line.split(",")
                if len(parts) >= 11:
                    ts = parts[1]
                    values = [float(x) for x in parts[2:]]
                    self.latest[key] = {"ts": ts, "vals": values}
                    self.counts[key] += 1
            except (ValueError, IndexError):
                pass
        return handler

    async def connect(self):
        print(f"\n  Scanning for {DEVICE_NAME}...")

        device = None
        for attempt in range(3):
            print(f"  Scan {attempt+1}/3...")
            devices = await BleakScanner.discover(timeout=8.0)
            for d in devices:
                if d.name == DEVICE_NAME:
                    device = d
                    print(f"  Found: {d.name} ({d.address})")
                    break
            if device:
                break

        if not device:
            print(f"  ERROR: {DEVICE_NAME} not found. Is ESP32 powered on?")
            return False

        print("  Connecting...")
        self.client = BleakClient(device.address)
        await self.client.connect()
        print("  Connected!")

        # Subscribe to all 3 characteristics
        await self.client.start_notify(CHAR_THIGH, self._make_handler("thigh"))
        await self.client.start_notify(CHAR_SHANK, self._make_handler("shank"))
        await self.client.start_notify(CHAR_FOOT,  self._make_handler("foot"))
        print("  Subscribed to THIGH, SHANK, FOOT channels")

        return True

    async def record(self):
        print("\n" + "=" * 60)
        print("  RECORDING - Press SPACE for event, Q to stop")
        print("=" * 60 + "\n")

        self._setup_kb()
        start = time.time()
        last_status = start

        try:
            while self.running:
                t0 = time.time()
                self._check_kb()

                any_data = any(v is not None for v in self.latest.values())
                if any_data:
                    # Get timestamp from whichever sensor has data
                    ts = ""
                    for k in ("thigh", "shank", "foot"):
                        if self.latest[k]:
                            ts = self.latest[k]["ts"]
                            break

                    row = [ts]

                    # Thigh: 9 vals
                    if self.latest["thigh"]:
                        row.extend(self.latest["thigh"]["vals"][:9])
                        self.latest["thigh"] = None
                    else:
                        row.extend([""] * 9)

                    # Shank: 9 vals
                    if self.latest["shank"]:
                        row.extend(self.latest["shank"]["vals"][:9])
                        self.latest["shank"] = None
                    else:
                        row.extend([""] * 9)

                    # Foot: 9 raw + 3 euler = 12 vals
                    if self.latest["foot"]:
                        fv = self.latest["foot"]["vals"]
                        row.extend(fv[:9])
                        row.extend(fv[9:12] if len(fv) >= 12 else [""] * 3)
                        self.latest["foot"] = None
                    else:
                        row.extend([""] * 12)

                    # Perturbation
                    if self.perturb_flag:
                        row.append(1)
                        self.perturb_flag = False
                    else:
                        row.append(0)

                    self.buffer.append(row)

                # Status every 2s
                if time.time() - last_status > 2:
                    el = time.time() - start
                    ev = sum(1 for r in self.buffer if r[-1] == 1)
                    c = self.counts
                    print(f"  [{el:.1f}s] T:{c['thigh']} S:{c['shank']} F:{c['foot']} | Events:{ev} | Rows:{len(self.buffer)}")
                    last_status = time.time()

                dt = time.time() - t0
                if dt < 0.01:
                    await asyncio.sleep(0.01 - dt)

        except KeyboardInterrupt:
            pass

        await self.stop()

    def _setup_kb(self):
        if sys.platform != "win32":
            import tty, termios
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def _cleanup_kb(self):
        if sys.platform != "win32" and hasattr(self, "_old"):
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def _check_kb(self):
        try:
            import select
            if sys.platform != "win32":
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        self.perturb_flag = True
                        if self.buffer:
                            self.buffer[-1][-1] = 1
                        n = sum(1 for r in self.buffer if r[-1] == 1)
                        print(f"\n  *** PERTURBATION #{n} ***\n")
                    elif ch.lower() == "q":
                        self.running = False
        except Exception:
            pass

    async def stop(self):
        print("\n  Stopping...")
        if self.client:
            try:
                await self.client.stop_notify(CHAR_THIGH)
                await self.client.stop_notify(CHAR_SHANK)
                await self.client.stop_notify(CHAR_FOOT)
                await self.client.disconnect()
            except Exception:
                pass
        self._cleanup_kb()
        self._save()

    def _save(self):
        if not self.buffer:
            print("  No data!")
            return

        fp = os.path.join(OUTPUT_DIR, f"{self.trial_name}.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            w.writerows(self.buffer)

        ev = sum(1 for r in self.buffer if r[-1] == 1)
        print(f"\n  {'=' * 50}")
        print(f"  SAVED: {fp}")
        print(f"  Rows: {len(self.buffer)}")
        print(f"  Events: {ev}")
        print(f"  Counts: T={self.counts['thigh']} S={self.counts['shank']} F={self.counts['foot']}")
        print(f"  {'=' * 50}\n")


async def quick_test():
    print("\n" + "=" * 60)
    print("  QUICK TEST (10 seconds)")
    print("=" * 60)

    logger = IMULogger("test")
    if not await logger.connect():
        return

    print("\n  Receiving for 10s...\n")
    start = time.time()
    while time.time() - start < 10:
        await asyncio.sleep(1)
        el = time.time() - start
        for k in ("thigh", "shank", "foot"):
            c = logger.counts[k]
            d = logger.latest[k]
            if d:
                preview = ", ".join(f"{v:.2f}" for v in d["vals"][:6])
                print(f"  [{el:.0f}s] {k.upper():6s} n={c:4d}: {preview}")
            else:
                print(f"  [{el:.0f}s] {k.upper():6s} n={c:4d}: --")

    print(f"\n  RESULTS:")
    for k in ("thigh", "shank", "foot"):
        c = logger.counts[k]
        hz = c / 10
        s = "OK" if c > 50 else "LOW" if c > 0 else "FAIL"
        print(f"    {k.upper():6s}: {c} ({hz:.0f}Hz) [{s}]")

    await logger.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", default=None)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        asyncio.run(quick_test())
    else:
        asyncio.run(main_record(args.trial))


async def main_record(trial):
    logger = IMULogger(trial)
    if not await logger.connect():
        return
    await logger.record()


if __name__ == "__main__":
    main()
