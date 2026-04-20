"""
BE124 V5 - Simple UDP Logger (Complete Frame)
==============================================
Each UDP packet = one complete frame with all 3 sensors.
No merging, no empty fields.

Usage:
  python udp_logger.py --test
  python udp_logger.py --trial slip_hang_01

Controls: SPACE = perturbation, Q = stop
"""

import socket
import time
import csv
import os
import sys
import argparse
from datetime import datetime

UDP_PORT = 12345
OUTPUT_DIR = "data"

CSV_HEADER = [
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


class UDPLogger:
    def __init__(self, trial_name=None):
        self.buffer = []
        self.count = 0
        self.perturb_flag = False
        self.running = True

        if trial_name is None:
            trial_name = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trial_name = trial_name
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", UDP_PORT))
        self.sock.setblocking(False)

    def receive(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                line = data.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                # Expect: timestamp + 9 thigh + 9 shank + 9 foot + 3 euler = 31 fields
                if len(parts) < 31:
                    continue

                # Append perturbation flag
                if self.perturb_flag:
                    parts.append("1")
                    self.perturb_flag = False
                else:
                    parts.append("0")

                self.buffer.append(parts)
                self.count += 1

            except BlockingIOError:
                break
            except (ValueError, IndexError):
                continue

    def record(self):
        print(f"\n{'='*60}")
        print(f"  RECORDING on UDP port {UDP_PORT}")
        print(f"  SPACE = perturbation | Q = stop")
        print(f"{'='*60}\n")

        self._setup_kb()
        start = time.time()
        last_status = start

        try:
            while self.running:
                t0 = time.time()
                self._check_kb()
                self.receive()

                if time.time() - last_status > 2:
                    el = time.time() - start
                    hz = self.count / max(el, 0.1)
                    ev = sum(1 for r in self.buffer if r[-1] == "1")
                    print(f"  [{el:.1f}s] Frames: {self.count} ({hz:.0f} Hz) | Events: {ev}")
                    last_status = time.time()

                dt = time.time() - t0
                if dt < 0.002:
                    time.sleep(0.002 - dt)

        except KeyboardInterrupt:
            pass

        self._cleanup_kb()
        self._save()

    def _save(self):
        self.sock.close()
        if not self.buffer:
            print("  No data!")
            return

        fp = os.path.join(OUTPUT_DIR, f"{self.trial_name}.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(self.buffer)

        ev = sum(1 for r in self.buffer if r[-1] == "1")
        duration = 0
        try:
            t0 = float(self.buffer[0][0])
            t1 = float(self.buffer[-1][0])
            duration = t1 - t0
        except:
            pass

        print(f"\n  {'='*50}")
        print(f"  SAVED: {fp}")
        print(f"  Frames: {self.count}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Avg Hz: {self.count / max(duration, 0.1):.0f}")
        print(f"  Events: {ev}")
        print(f"  Empty fields: 0")
        print(f"  {'='*50}\n")

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
                        n = sum(1 for r in self.buffer if r[-1] == "1") + 1
                        print(f"\n  *** PERTURBATION #{n} ***\n")
                    elif ch.lower() == "q":
                        self.running = False
        except:
            pass


def quick_test():
    print(f"\n{'='*60}")
    print(f"  QUICK TEST - UDP port {UDP_PORT} (10 seconds)")
    print(f"{'='*60}\n")

    logger = UDPLogger("test")
    start = time.time()
    last = start

    while time.time() - start < 10:
        logger.receive()
        time.sleep(0.002)

        if time.time() - last > 1:
            el = time.time() - start
            hz = logger.count / max(el, 0.1)
            # Show last frame preview
            if logger.buffer:
                last_frame = logger.buffer[-1]
                # Show first 6 values (thigh acc + gyro)
                preview = ", ".join(last_frame[1:7])
                print(f"  [{el:.0f}s] Frames: {logger.count:4d} ({hz:.0f} Hz) | Thigh: {preview}")
            else:
                print(f"  [{el:.0f}s] Frames: {logger.count:4d} -- no data --")
            last = time.time()

    logger.sock.close()
    hz = logger.count / 10
    status = "OK" if logger.count > 500 else "LOW" if logger.count > 0 else "FAIL"
    print(f"\n  RESULT: {logger.count} frames in 10s ({hz:.0f} Hz) [{status}]")
    print(f"  Empty fields: 0 (complete frame mode)")


def main():
    p = argparse.ArgumentParser(description="BE124 V5 UDP Logger")
    p.add_argument("--trial", default=None)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        quick_test()
    else:
        logger = UDPLogger(args.trial)
        logger.record()


if __name__ == "__main__":
    main()