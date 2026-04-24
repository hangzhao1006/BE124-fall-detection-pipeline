#!/usr/bin/env python3
"""
BE124 Data Cleaning Pipeline
==============================
Checks for anomalies, fixes units, and produces clean CSV files.

Usage:
  # Clean a single file
  python clean_data.py data/slip_hang_01.csv

  # Clean all files in data/
  python clean_data.py data/

  # Clean + apply low-pass filter
  python clean_data.py data/slip_hang_01.csv --lowpass

  # Custom thresholds
  python clean_data.py data/ --acc-max 60 --gyro-max 25

Output:
  Cleaned files saved to cleaned/ folder
  Report printed to terminal

Requirements:
  pip install pandas numpy scipy matplotlib
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import argparse
import sys


# ============================================================
# Default thresholds
# ============================================================
ACC_MAX = 50.0      # m/s², anything above this is anomaly
GYRO_MAX = 20.0     # rad/s, for ICM-20948 (already converted)
GYRO_MAX_DEG = 1145  # °/s, for unconverted BNO055 (20 rad/s * 57.3)
MAG_MAX = 5000.0    # µT, raised high - mag spikes from nearby metal are common indoors
EULER_MAX = 360.0   # degrees - Euler angles are NOT checked for anomalies (0-360 is valid)

# Column groups
THIGH_ACC = ['thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z']
THIGH_GYRO = ['thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z']
THIGH_MAG = ['thigh_mag_x', 'thigh_mag_y', 'thigh_mag_z']

SHANK_ACC = ['shank_acc_x', 'shank_acc_y', 'shank_acc_z']
SHANK_GYRO = ['shank_gyro_x', 'shank_gyro_y', 'shank_gyro_z']
SHANK_MAG = ['shank_mag_x', 'shank_mag_y', 'shank_mag_z']

FOOT_ACC = ['foot_acc_x', 'foot_acc_y', 'foot_acc_z']
FOOT_GYRO = ['foot_gyro_x', 'foot_gyro_y', 'foot_gyro_z']
FOOT_MAG = ['foot_mag_x', 'foot_mag_y', 'foot_mag_z']
FOOT_EULER = ['foot_euler_x', 'foot_euler_y', 'foot_euler_z']

ALL_ACC = THIGH_ACC + SHANK_ACC + FOOT_ACC
ALL_GYRO = THIGH_GYRO + SHANK_GYRO + FOOT_GYRO
ALL_MAG = THIGH_MAG + SHANK_MAG + FOOT_MAG


# ============================================================
# Unit check & conversion
# ============================================================
def check_and_fix_gyro_units(df):
    """
    Detect if BNO055 gyro is in °/s and convert to rad/s.
    ICM-20948 gyro range: typically ±5 rad/s during walking
    BNO055 unconverted: typically ±300 °/s during walking
    """
    issues = []

    # Check FOOT gyro range vs THIGH gyro range
    foot_range = 0
    thigh_range = 0

    for col in FOOT_GYRO:
        if col in df.columns:
            vals = df[col].dropna().abs()
            if len(vals) > 0:
                foot_range = max(foot_range, vals.quantile(0.99))

    for col in THIGH_GYRO:
        if col in df.columns:
            vals = df[col].dropna().abs()
            if len(vals) > 0:
                thigh_range = max(thigh_range, vals.quantile(0.99))

    if thigh_range > 0 and foot_range > thigh_range * 10:
        # FOOT gyro is likely in °/s, convert to rad/s
        issues.append(f"BNO055 gyro in °/s (range {foot_range:.1f}), converting to rad/s")
        for col in FOOT_GYRO:
            if col in df.columns:
                df[col] = df[col] / 57.2958
    elif foot_range > 0 and thigh_range > 0:
        ratio = foot_range / thigh_range
        if ratio < 5:
            issues.append(f"Gyro units appear consistent (FOOT/THIGH ratio: {ratio:.1f})")
        else:
            issues.append(f"WARNING: Gyro ratio suspicious (FOOT/THIGH: {ratio:.1f}), check manually")

    return df, issues


# ============================================================
# Anomaly detection
# ============================================================
def detect_anomalies(df, acc_max=ACC_MAX, gyro_max=GYRO_MAX, mag_max=MAG_MAX):
    """
    Find data points outside physically reasonable ranges.
    Returns a dict with anomaly info.
    """
    anomalies = {}
    total_anomalies = 0

    # Check accelerometer
    for col in ALL_ACC:
        if col in df.columns:
            mask = df[col].abs() > acc_max
            count = mask.sum()
            if count > 0:
                anomalies[col] = {
                    'count': count,
                    'max_val': df[col].abs().max(),
                    'indices': df.index[mask].tolist()
                }
                total_anomalies += count

    # Check gyroscope
    for col in ALL_GYRO:
        if col in df.columns:
            mask = df[col].abs() > gyro_max
            count = mask.sum()
            if count > 0:
                anomalies[col] = {
                    'count': count,
                    'max_val': df[col].abs().max(),
                    'indices': df.index[mask].tolist()
                }
                total_anomalies += count

    # Magnetometer and Euler angles are NOT checked for anomalies:
    # - Mag spikes from nearby metal/phones are common and expected indoors
    # - Euler angles (0-360°) are always in valid range
    # - Neither is critical for the fall prediction model

    # Check for constant values (sensor stuck)
    for col in ALL_ACC + ALL_GYRO:
        if col in df.columns:
            # Check if 50+ consecutive identical values
            diffs = df[col].diff()
            consecutive_zeros = (diffs == 0).astype(int)
            groups = consecutive_zeros.groupby((consecutive_zeros != consecutive_zeros.shift()).cumsum())
            max_run = groups.sum().max()
            if max_run > 50:
                anomalies[f'{col}_stuck'] = {
                    'count': int(max_run),
                    'max_val': None,
                    'indices': [],
                    'note': f'Sensor appears stuck for {int(max_run)} consecutive frames'
                }
                total_anomalies += 1

    # Check for NaN/empty values
    for col in df.columns:
        if col in ['timestamp', 'time_s', 'perturbation_event']:
            continue
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            anomalies[f'{col}_nan'] = {
                'count': nan_count,
                'max_val': None,
                'indices': df.index[df[col].isna()].tolist()[:10],  # first 10
                'note': f'{nan_count} NaN values'
            }
            total_anomalies += nan_count

    # Check timestamp consistency
    if 'timestamp' in df.columns:
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        diffs = ts.diff().dropna()

        # Negative time jumps
        neg = (diffs < 0).sum()
        if neg > 0:
            anomalies['timestamp_negative'] = {
                'count': neg,
                'max_val': diffs.min(),
                'indices': [],
                'note': f'{neg} negative time jumps detected'
            }
            total_anomalies += neg

        # Large time gaps (> 50ms at 144Hz, expected ~7ms)
        large_gaps = (diffs > 0.05).sum()
        if large_gaps > 0:
            anomalies['timestamp_gaps'] = {
                'count': large_gaps,
                'max_val': diffs.max(),
                'indices': [],
                'note': f'{large_gaps} gaps > 50ms (max: {diffs.max()*1000:.1f}ms)'
            }

    return anomalies, total_anomalies


# ============================================================
# Fix anomalies
# ============================================================
def fix_anomalies(df, acc_max=ACC_MAX, gyro_max=GYRO_MAX, mag_max=MAG_MAX):
    """
    Replace anomalous values with interpolated values.
    """
    fixed_count = 0

    # Fix accelerometer outliers
    for col in ALL_ACC:
        if col in df.columns:
            mask = df[col].abs() > acc_max
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = np.nan
                fixed_count += count

    # Fix gyroscope outliers
    for col in ALL_GYRO:
        if col in df.columns:
            mask = df[col].abs() > gyro_max
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = np.nan
                fixed_count += count

    # Magnetometer is NOT fixed - spikes are from environment, not sensor errors
    # Euler angles are NOT fixed - always in valid range

    # Interpolate NaN values
    if fixed_count > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skip = ['timestamp', 'perturbation_event']
        for col in numeric_cols:
            if col not in skip:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')

    return df, fixed_count


# ============================================================
# Optional low-pass filter
# ============================================================
def apply_lowpass(df, cutoff_hz=20, order=4):
    """Apply Butterworth low-pass filter."""
    duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
    fs = len(df) / duration

    if fs < cutoff_hz * 2.5:
        print(f"  WARNING: Sample rate ({fs:.0f}Hz) too low for {cutoff_hz}Hz cutoff")
        return df

    b, a = butter(order, cutoff_hz / (fs / 2), btype='low')
    skip = ['timestamp', 'time_s', 'perturbation_event']

    filtered_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in skip:
            valid = df[col].dropna()
            if len(valid) > 3 * max(len(a), len(b)):
                df[col] = filtfilt(b, a, df[col].ffill().bfill())
                filtered_count += 1

    print(f"  Low-pass filter: {cutoff_hz}Hz cutoff applied to {filtered_count} columns")
    return df


# ============================================================
# Main cleaning pipeline
# ============================================================
def clean_file(filepath, acc_max=ACC_MAX, gyro_max=GYRO_MAX, do_lowpass=False):
    """Full cleaning pipeline for one CSV file."""
    name = Path(filepath).stem
    print(f"\n{'='*60}")
    print(f"  Cleaning: {filepath}")
    print(f"{'='*60}")

    # Load
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

    original_rows = len(df)
    duration = df['time_s'].iloc[-1]
    hz = original_rows / duration if duration > 0 else 0

    print(f"\n  Raw data: {original_rows} rows, {duration:.1f}s, {hz:.0f}Hz")

    # Step 1: Check and fix gyro units
    print(f"\n  --- Step 1: Gyro unit check ---")
    df, unit_issues = check_and_fix_gyro_units(df)
    for issue in unit_issues:
        print(f"  {issue}")

    # Step 2: Detect anomalies
    print(f"\n  --- Step 2: Anomaly detection ---")
    anomalies, total = detect_anomalies(df, acc_max, gyro_max)

    if total == 0:
        print(f"  No anomalies found!")
    else:
        print(f"  Found {total} anomalies:")
        for col, info in anomalies.items():
            if 'note' in info:
                print(f"    {col}: {info['note']}")
            else:
                print(f"    {col}: {info['count']} outliers (max |value|: {info['max_val']:.2f})")

    # Step 3: Fix anomalies
    print(f"\n  --- Step 3: Fixing anomalies ---")
    df, fixed = fix_anomalies(df, acc_max, gyro_max)
    print(f"  Replaced {fixed} outlier values with interpolation")

    # Step 4: Optional low-pass filter
    if do_lowpass:
        print(f"\n  --- Step 4: Low-pass filter ---")
        df = apply_lowpass(df)

    # Step 5: Verify
    print(f"\n  --- Final verification ---")
    anomalies_after, total_after = detect_anomalies(df, acc_max, gyro_max)
    nan_count = df.isna().sum().sum()
    print(f"  Remaining anomalies: {total_after}")
    print(f"  Remaining NaN: {nan_count}")
    print(f"  Rows: {len(df)}")

    # Print value ranges
    print(f"\n  --- Value ranges (cleaned) ---")
    for sensor in ['thigh', 'shank', 'foot']:
        acc_cols = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
        gyro_cols = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']

        acc_vals = df[acc_cols].values.flatten()
        acc_vals = acc_vals[~np.isnan(acc_vals)]
        gyro_vals = df[gyro_cols].values.flatten()
        gyro_vals = gyro_vals[~np.isnan(gyro_vals)]

        if len(acc_vals) > 0 and len(gyro_vals) > 0:
            print(f"  {sensor.upper():6s} acc: [{acc_vals.min():.2f}, {acc_vals.max():.2f}] m/s² | "
                  f"gyro: [{gyro_vals.min():.2f}, {gyro_vals.max():.2f}] rad/s")

    # Save
    out_dir = Path('cleaned')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{name}_clean.csv'

    # Drop helper column
    if 'time_s' in df.columns:
        df = df.drop(columns=['time_s'])

    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return {
        'file': name,
        'original_rows': original_rows,
        'hz': hz,
        'duration': duration,
        'anomalies_found': total,
        'anomalies_fixed': fixed,
        'unit_converted': any('converting' in i for i in unit_issues),
    }


def clean_batch(data_dir, acc_max=ACC_MAX, gyro_max=GYRO_MAX, do_lowpass=False):
    """Clean all CSVs in a directory."""
    csv_files = sorted(Path(data_dir).glob('*.csv'))
    if not csv_files:
        print(f"  No CSV files in {data_dir}")
        return

    print(f"\n  Found {len(csv_files)} files in {data_dir}")

    results = []
    for fp in csv_files:
        result = clean_file(str(fp), acc_max, gyro_max, do_lowpass)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"  {'File':<30s} {'Rows':>6s} {'Hz':>5s} {'Anomalies':>9s} {'Fixed':>6s} {'Unit':>5s}")
    print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*9} {'-'*6} {'-'*5}")
    for r in results:
        unit = 'YES' if r['unit_converted'] else '-'
        print(f"  {r['file']:<30s} {r['original_rows']:>6d} {r['hz']:>5.0f} "
              f"{r['anomalies_found']:>9d} {r['anomalies_fixed']:>6d} {unit:>5s}")


def main():
    parser = argparse.ArgumentParser(description='BE124 Data Cleaning Pipeline')
    parser.add_argument('paths', nargs='+', help='CSV file(s) or directory')
    parser.add_argument('--lowpass', action='store_true', help='Apply 20Hz low-pass filter')
    parser.add_argument('--acc-max', type=float, default=ACC_MAX,
                        help=f'Max acceleration threshold (default: {ACC_MAX})')
    parser.add_argument('--gyro-max', type=float, default=GYRO_MAX,
                        help=f'Max gyro threshold in rad/s (default: {GYRO_MAX})')

    args = parser.parse_args()

    for p in args.paths:
        path = Path(p)
        if path.is_file():
            clean_file(str(path), args.acc_max, args.gyro_max, args.lowpass)
        elif path.is_dir():
            clean_batch(str(path), args.acc_max, args.gyro_max, args.lowpass)
        else:
            print(f"  Not found: {p}")


if __name__ == '__main__':
    main()