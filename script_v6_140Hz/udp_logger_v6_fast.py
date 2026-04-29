"""
BE124 V6.2 - Optimized Binary UDP Logger
==========================================
Uses a dedicated thread for receiving UDP packets.
Receive thread does NOTHING except read from socket and put into queue.
Main thread handles keyboard, display, and CSV saving.

Usage:
  python udp_logger_v6_fast.py --test
  python udp_logger_v6_fast.py --trial trip_hang_01
"""

import socket
import struct
import time
import csv
import os
import sys
import argparse
import threading
import queue
from datetime import datetime

UDP_PORT = 12345
OUTPUT_DIR = "data"
MAGIC = 0xBE124DAA

FRAME_FMT = "<IIHH30f"
FRAME_SIZE = struct.calcsize(FRAME_FMT)

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


class FastUDPLogger:
    def __init__(self, trial_name=None):
        self.buffer = []
        self.count = 0
        self.dropped = 0
        self.last_frame_id = -1
        self.perturb_flag = False
        self.event_count = 0
        self.running = True

        if trial_name is None:
            trial_name = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trial_name = trial_name
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Raw packet queue - receive thread puts raw bytes here
        self.packet_queue = queue.Queue(maxsize=50000)

        # Socket with large buffer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", UDP_PORT))
        self.sock.settimeout(0.1)  # 100ms timeout for clean shutdown

        print(f"  Expected packet size: {FRAME_SIZE} bytes")

    def _receive_thread(self):
        """Dedicated thread: ONLY receives packets. No processing."""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(256)
                if len(data) == FRAME_SIZE:
                    self.packet_queue.put(data, block=False)
            except socket.timeout:
                continue
            except queue.Full:
                pass  # drop if queue full (shouldn't happen)
            except OSError:
                break

    def _process_packets(self):
        """Process all queued packets."""
        while True:
            try:
                data = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            try:
                vals = struct.unpack(FRAME_FMT, data)
            except struct.error:
                continue

            if vals[0] != MAGIC:
                continue

            epoch_sec = vals[1]
            epoch_ms = vals[2]
            frame_id = vals[3]
            floats = vals[4:]

            # Check dropped
            if self.last_frame_id >= 0:
                expected = (self.last_frame_id + 1) & 0xFFFF
                if frame_id != expected:
                    gap = (frame_id - self.last_frame_id) & 0xFFFF
                    if gap < 1000:
                        self.dropped += gap - 1
            self.last_frame_id = frame_id

            ts = f"{epoch_sec}.{epoch_ms:03d}"
            row = [ts]
            row.extend([f"{v:.4f}" for v in floats])

            if self.perturb_flag:
                row.append("1")
                self.perturb_flag = False
            else:
                row.append("0")

            self.buffer.append(row)
            self.count += 1

    def record(self):
        print(f"\n{'='*60}")
        print(f"  RECORDING (Binary UDP, port {UDP_PORT})")
        print(f"  SPACE = perturbation | Q = stop")
        print(f"{'='*60}\n")

        # Start receive thread
        rx_thread = threading.Thread(target=self._receive_thread, daemon=True)
        rx_thread.start()

        self._setup_kb()
        start = time.time()
        last_status = start
        last_kb = start

        try:
            while self.running:
                # Process queued packets
                self._process_packets()

                now = time.time()

                # Keyboard check every 100ms
                if now - last_kb > 0.1:
                    self._check_kb()
                    last_kb = now

                # Status every 2s
                if now - last_status > 2:
                    el = now - start
                    hz = self.count / max(el, 0.1)
                    print(f"  [{el:.1f}s] Frames: {self.count} ({hz:.0f} Hz) | Dropped: {self.dropped} | Events: {self.event_count}")
                    last_status = now

        except KeyboardInterrupt:
            pass

        self.running = False
        rx_thread.join(timeout=1)
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

        duration = 0
        try:
            duration = float(self.buffer[-1][0]) - float(self.buffer[0][0])
        except:
            pass

        print(f"\n  {'='*50}")
        print(f"  SAVED: {fp}")
        print(f"  Frames: {self.count}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Avg Hz: {self.count / max(duration, 0.1):.0f}")
        print(f"  Dropped: {self.dropped}")
        print(f"  Events: {self.event_count}")
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
                        self.event_count += 1
                        print(f"\n  *** PERTURBATION #{self.event_count} ***\n")
                    elif ch.lower() == "q":
                        self.running = False
        except:
            pass


def quick_test():
    print(f"\n{'='*60}")
    print(f"  QUICK TEST - Threaded UDP port {UDP_PORT} (10 seconds)")
    print(f"  Packet size: {FRAME_SIZE} bytes")
    print(f"{'='*60}\n")

    logger = FastUDPLogger("test")

    rx_thread = threading.Thread(target=logger._receive_thread, daemon=True)
    rx_thread.start()

    start = time.time()
    last = start

    while time.time() - start < 10:
        logger._process_packets()
        time.sleep(0.001)

        if time.time() - last > 1:
            el = time.time() - start
            hz = logger.count / max(el, 0.1)
            if logger.buffer:
                r = logger.buffer[-1]
                preview = ", ".join(r[1:7])
                print(f"  [{el:.0f}s] Frames: {logger.count:4d} ({hz:.0f} Hz) | Drop: {logger.dropped} | Thigh: {preview}")
            else:
                print(f"  [{el:.0f}s] Frames: {logger.count:4d} -- no data --")
            last = time.time()

    logger.running = False
    rx_thread.join(timeout=1)
    logger.sock.close()

    hz = logger.count / 10
    status = "OK" if logger.count > 500 else "LOW" if logger.count > 0 else "FAIL"
    print(f"\n  RESULT: {logger.count} frames in 10s ({hz:.0f} Hz) [{status}]")
    print(f"  Dropped frames: {logger.dropped}")


def main():
    p = argparse.ArgumentParser(description="BE124 V6.2 Fast Binary UDP Logger")
    p.add_argument("--trial", default=None)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        quick_test()
    else:
        logger = FastUDPLogger(args.trial)
        logger.record()


if __name__ == "__main__":
    main()