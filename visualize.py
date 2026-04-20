#!/usr/bin/env python3
"""
BE124 Ankle Exoskeleton - Local Data Visualization & Analysis Pipeline
======================================================================
Place this script in your BE124_Data_Collection folder.

Project structure (recommended):
BE124_Data_Collection/
├── firmware/                    # V4 WiFi UDP firmware (PCA9548A + 3 IMUs)
│   └── firmware.ino
├── firmware_1/                  # Backup/alternative firmware
│   └── firmware_1.ino
├── firmware_icm20948/           # Standalone ICM-20948 firmware (for testing)
├── data/                        # Raw CSV files from udp_logger.py
│   ├── slip_hang_01.csv
│   ├── slip_hang_02.csv
│   ├── trip_hang_01.csv
│   ├── normal_hang_01.csv
│   └── ...
├── processed/                   # Interpolated & cleaned data (auto-generated)
├── figures/                     # Saved plots (auto-generated)
├── models/                      # Trained model checkpoints (later)
├── udp_logger.py                # Data collection script
├── ble_logger.py                # BLE version (backup)
├── visualize.py                 # <-- THIS FILE
└── README.md


Usage:
  # Visualize a single trial
  python visualize.py data/slip_hang_01.csv

  # Visualize and save figures
  python visualize.py data/slip_hang_01.csv --save

  # Compare multiple trials
  python visualize.py data/slip_hang_01.csv data/normal_hang_01.csv --compare

  # Batch process all CSVs in data/
  python visualize.py data/ --batch


  # 看单个trial（6个图：XYZ加速度、XYZ角速度、幅值、关节角度、频谱、降采样对比）
  python visualize.py data/slip_hang_01.csv

  # 看 + 保存图片到 figures/
  python visualize.py data/slip_hang_01.csv --save

  # 批量处理 data/ 下所有 CSV
  python visualize.py data/ --batch

  # 叠加对比多个 trial
  python visualize.py data/slip_01.csv data/normal_01.csv --compare

Requirements:
  pip3 install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pathlib import Path
import argparse
import sys
import os


# ============================================================
# Color scheme
# ============================================================
COLORS = {
    'thigh': '#534AB7',   # purple
    'shank': '#1D9E75',   # teal
    'foot':  '#D85A30',   # coral
}

SENSOR_COLS = {
    'thigh': {
        'acc': ['thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z'],
        'gyro': ['thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z'],
        'mag': ['thigh_mag_x', 'thigh_mag_y', 'thigh_mag_z'],
    },
    'shank': {
        'acc': ['shank_acc_x', 'shank_acc_y', 'shank_acc_z'],
        'gyro': ['shank_gyro_x', 'shank_gyro_y', 'shank_gyro_z'],
        'mag': ['shank_mag_x', 'shank_mag_y', 'shank_mag_z'],
    },
    'foot': {
        'acc': ['foot_acc_x', 'foot_acc_y', 'foot_acc_z'],
        'gyro': ['foot_gyro_x', 'foot_gyro_y', 'foot_gyro_z'],
        'mag': ['foot_mag_x', 'foot_mag_y', 'foot_mag_z'],
        'euler': ['foot_euler_x', 'foot_euler_y', 'foot_euler_z'],
    },
}


# ============================================================
# Data loading & preprocessing
# ============================================================
def load_trial(filepath):
    """Load a trial CSV and compute relative time."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df


def compute_magnitudes(df):
    """Compute acceleration and gyroscope magnitudes for each sensor."""
    for sensor, cols in SENSOR_COLS.items():
        # Acceleration magnitude
        acc_cols = cols['acc']
        if all(c in df.columns for c in acc_cols):
            df[f'{sensor}_acc_mag'] = np.sqrt(
                df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2
            )

        # Gyroscope magnitude
        gyro_cols = cols['gyro']
        if all(c in df.columns for c in gyro_cols):
            df[f'{sensor}_gyro_mag'] = np.sqrt(
                df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2
            )
    return df


def interpolate_missing(df, method='linear'):
    """Fill missing values via interpolation."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['timestamp', 'time_s', 'perturbation_event']:
            df[col] = df[col].interpolate(method=method, limit_direction='both')
    return df


def resample_to_hz(df, target_hz):
    """Resample data to a target frequency for downsampling experiments."""
    duration = df['time_s'].iloc[-1]
    n_samples = int(duration * target_hz)
    new_times = np.linspace(0, duration, n_samples)

    resampled = pd.DataFrame({'time_s': new_times})
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['timestamp', 'time_s']]

    for col in numeric_cols:
        valid = df[['time_s', col]].dropna()
        if len(valid) > 2:
            f = interp1d(valid['time_s'], valid[col], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            resampled[col] = f(new_times)
        else:
            resampled[col] = np.nan

    return resampled


def apply_lowpass(df, cutoff_hz=20, fs=None, order=4):
    if fs is None:
        duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
        fs = len(df) / duration

    b, a = butter(order, cutoff_hz / (fs / 2), btype='low')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skip = ['timestamp', 'time_s', 'perturbation_event']

    for col in numeric_cols:
        if col not in skip:
            valid = df[col].dropna()
            if len(valid) > 3 * max(len(a), len(b)):
                df[col] = filtfilt(b, a, df[col].fillna(method='ffill').fillna(method='bfill'))
    return df


# ============================================================
# Visualization functions
# ============================================================
def print_summary(df, filepath):
    """Print data summary statistics."""
    duration = df['time_s'].iloc[-1]
    n_rows = len(df)
    hz = n_rows / duration if duration > 0 else 0

    events = df['perturbation_event'].sum() if 'perturbation_event' in df.columns else 0

    missing = {}
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_x'
        if col in df.columns:
            missing[sensor] = df[col].isna().sum()

    print(f"\n{'='*60}")
    print(f"  File: {filepath}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Rows: {n_rows:,}")
    print(f"  Sample rate: ~{hz:.0f} Hz")
    print(f"  Perturbation events: {events}")
    print(f"  Missing data: {missing}")
    print(f"{'='*60}\n")


def plot_overview(df, title="", save_path=None):
    """Plot 6-panel overview: acceleration + gyroscope for each sensor."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(f'IMU Data Overview — {title}', fontsize=14, fontweight='bold')

    t = df['time_s']

    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        color = COLORS[sensor]

        # Acceleration XYZ
        ax = axes[i, 0]
        for j, axis_label in enumerate(['x', 'y', 'z']):
            col = f'{sensor}_acc_{axis_label}'
            if col in df.columns:
                alpha = [1.0, 0.6, 0.4][j]
                ax.plot(t, df[col], linewidth=0.8, alpha=alpha,
                        label=axis_label.upper(), color=color)
        ax.set_ylabel('m/s²')
        ax.set_title(f'{sensor.upper()} acceleration', fontsize=11, color=color)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        # Gyroscope XYZ
        ax = axes[i, 1]
        for j, axis_label in enumerate(['x', 'y', 'z']):
            col = f'{sensor}_gyro_{axis_label}'
            if col in df.columns:
                alpha = [1.0, 0.6, 0.4][j]
                ax.plot(t, df[col], linewidth=0.8, alpha=alpha,
                        label=axis_label.upper(), color=color)
        ax.set_ylabel('rad/s')
        ax.set_title(f'{sensor.upper()} gyroscope', fontsize=11, color=color)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Mark perturbation events
    if 'perturbation_event' in df.columns:
        events = df[df['perturbation_event'] == 1]
        for _, row in events.iterrows():
            for ax_row in axes:
                for ax in ax_row:
                    ax.axvline(x=row['time_s'], color='red', alpha=0.5,
                               linewidth=1, linestyle='--')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_magnitudes(df, title="", save_path=None):
    """Plot acceleration and gyroscope magnitudes for all sensors."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Signal Magnitudes — {title}', fontsize=14, fontweight='bold')

    t = df['time_s']

    # Acceleration magnitude
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            axes[0].plot(t, df[col], linewidth=0.8, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)
    axes[0].set_ylabel('Acceleration magnitude (m/s²)')
    axes[0].axhline(y=9.81, color='gray', linestyle=':', alpha=0.5, label='Gravity')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Gyroscope magnitude
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns:
            axes[1].plot(t, df[col], linewidth=0.8, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)
    axes[1].set_ylabel('Gyroscope magnitude (rad/s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    # Mark perturbation events
    if 'perturbation_event' in df.columns:
        events = df[df['perturbation_event'] == 1]
        for _, row in events.iterrows():
            for ax in axes:
                ax.axvline(x=row['time_s'], color='red', alpha=0.5,
                           linewidth=1.5, linestyle='--', label='_')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_joint_angles(df, title="", save_path=None):
    """Estimate and plot knee and ankle joint angles from Euler/roll data."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    fig.suptitle(f'Estimated Joint Angles — {title}', fontsize=14, fontweight='bold')

    t = df['time_s']

    # If we have foot euler angles from BNO055
    if 'foot_euler_y' in df.columns:
        ax.plot(t, df['foot_euler_y'], linewidth=1, color=COLORS['foot'],
                label='Foot roll (BNO055 Euler Y)', alpha=0.8)
    if 'foot_euler_z' in df.columns:
        ax.plot(t, df['foot_euler_z'], linewidth=1, color=COLORS['foot'],
                label='Foot pitch (BNO055 Euler Z)', alpha=0.6, linestyle='--')

    ax.set_ylabel('Angle (degrees)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Mark perturbation events
    if 'perturbation_event' in df.columns:
        events = df[df['perturbation_event'] == 1]
        for _, row in events.iterrows():
            ax.axvline(x=row['time_s'], color='red', alpha=0.5,
                       linewidth=1.5, linestyle='--')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_frequency_spectrum(df, sensor='thigh', axis='acc_x', title="", save_path=None):
    """Plot FFT frequency spectrum to check signal content."""
    col = f'{sensor}_{axis}'
    if col not in df.columns:
        print(f"  Column {col} not found")
        return

    data = df[col].dropna().values
    if len(data) < 64:
        print(f"  Not enough data for FFT")
        return

    duration = df['time_s'].iloc[-1]
    fs = len(data) / duration

    fft_vals = np.abs(np.fft.rfft(data - np.mean(data)))
    freqs = np.fft.rfftfreq(len(data), d=1/fs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(freqs, fft_vals, linewidth=0.8, color=COLORS[sensor])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Frequency Spectrum — {sensor.upper()} {axis} — {title}', fontsize=12)
    ax.set_xlim(0, fs/2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_downsample_comparison(df, title="", save_path=None):
    duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
    original_hz = int(len(df) / duration)
    rates = [original_hz, 100, 80, 50]

    fig, axes = plt.subplots(len(rates), 1, figsize=(16, 3 * len(rates)), sharex=True)
    fig.suptitle(f'Downsampling Comparison — Thigh Acc Magnitude — {title}',
                 fontsize=14, fontweight='bold')

    compute_magnitudes(df)

    for i, hz in enumerate(rates):
        resampled = resample_to_hz(df, hz)
        compute_magnitudes(resampled)
        axes[i].plot(resampled['time_s'], resampled['thigh_acc_mag'],
                     linewidth=0.8, color=COLORS['thigh'])
        axes[i].set_ylabel('m/s²')
        axes[i].set_title(f'{hz} Hz ({len(resampled)} samples)', fontsize=11)
        axes[i].axhline(y=9.81, color='gray', linestyle=':', alpha=0.3)
        axes[i].grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def compare_trials(filepaths, sensor='thigh', metric='acc_mag'):
    """Overlay multiple trials for comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    col = f'{sensor}_{metric}'

    colors = plt.cm.viridis(np.linspace(0, 1, len(filepaths)))

    for i, fp in enumerate(filepaths):
        df = load_trial(fp)
        df = interpolate_missing(df)
        compute_magnitudes(df)
        if col in df.columns:
            label = Path(fp).stem
            ax.plot(df['time_s'], df[col], linewidth=0.8, color=colors[i],
                    label=label, alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(col)
    ax.set_title(f'Trial Comparison — {sensor.upper()} {metric}', fontsize=14)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def process_single(filepath, save=False):
    """Full visualization pipeline for a single trial."""
    df = load_trial(filepath)
    name = Path(filepath).stem

    # Summary
    print_summary(df, filepath)

    # Interpolate
    df = interpolate_missing(df)
    compute_magnitudes(df)

    # Create output dirs
    fig_dir = Path('figures')
    proc_dir = Path('processed')
    fig_dir.mkdir(exist_ok=True)
    proc_dir.mkdir(exist_ok=True)

    # Save processed CSV
    proc_path = proc_dir / f'{name}_processed.csv'
    df.to_csv(proc_path, index=False)
    print(f"  Processed data saved: {proc_path}")

    # Plots
    save_prefix = fig_dir / name if save else None

    plot_overview(df, title=name,
                  save_path=f'{save_prefix}_overview.png' if save else None)
    plot_magnitudes(df, title=name,
                    save_path=f'{save_prefix}_magnitudes.png' if save else None)
    plot_joint_angles(df, title=name,
                      save_path=f'{save_prefix}_joints.png' if save else None)
    plot_frequency_spectrum(df, sensor='thigh', axis='acc_z', title=name,
                            save_path=f'{save_prefix}_fft_thigh.png' if save else None)

    # Downsampling comparison
    print("\n  Downsampling comparison:")
    plot_downsample_comparison(df, title=name,
                           save_path=f'{save_prefix}_downsample.png' if save else None)


def process_batch(data_dir, save=False):
    """Process all CSVs in a directory."""
    csv_files = sorted(Path(data_dir).glob('*.csv'))
    if not csv_files:
        print(f"  No CSV files found in {data_dir}")
        return

    print(f"\n  Found {len(csv_files)} trial(s) in {data_dir}")
    for fp in csv_files:
        print(f"\n{'='*60}")
        process_single(str(fp), save=save)


def main():
    parser = argparse.ArgumentParser(
        description='BE124 IMU Data Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py data/slip_hang_01.csv          # View single trial
  python visualize.py data/slip_hang_01.csv --save   # View + save figures
  python visualize.py data/ --batch                  # Process all CSVs
  python visualize.py data/slip_01.csv data/normal_01.csv --compare
        """)

    parser.add_argument('paths', nargs='+', help='CSV file(s) or directory')
    parser.add_argument('--save', action='store_true',
                        help='Save figures to figures/ directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process all CSVs in directory')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple trials')

    args = parser.parse_args()

    if args.compare and len(args.paths) > 1:
        compare_trials(args.paths)
    elif args.batch:
        for p in args.paths:
            if Path(p).is_dir():
                process_batch(p, save=args.save)
    else:
        for p in args.paths:
            if Path(p).is_file():
                process_single(p, save=args.save)
            elif Path(p).is_dir():
                process_batch(p, save=args.save)
            else:
                print(f"  Not found: {p}")


if __name__ == '__main__':
    main()
