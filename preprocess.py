#!/usr/bin/env python3
"""
BE124 Data Preprocessing Pipeline
===================================
1. Auto-label perturbation timestamps using energy spike detection
2. Resample all trials to uniform Hz
3. Sliding window segmentation
4. Train/val/test split (leave-one-subject-out)
5. Save as numpy arrays ready for model training

Usage:
  python preprocess.py --data-dir data/ --output-dir dataset/
  python preprocess.py --data-dir data/ --output-dir dataset/ --target-hz 100 --window-ms 500 --stride-ms 250

Requirements:
  pip install pandas numpy scipy scikit-learn
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pathlib import Path
import argparse
import json
import os


# ============================================================
# Config
# ============================================================
FEATURE_COLS = [
    'thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z',
    'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z',
    'shank_acc_x', 'shank_acc_y', 'shank_acc_z',
    'shank_gyro_x', 'shank_gyro_y', 'shank_gyro_z',
    'foot_acc_x', 'foot_acc_y', 'foot_acc_z',
    'foot_gyro_x', 'foot_gyro_y', 'foot_gyro_z',
]
# 18 features: 3 sensors × (3 acc + 3 gyro)
# Magnetometer and Euler excluded from model input
# (mag is noisy indoors, euler only available for foot)

N_FEATURES = len(FEATURE_COLS)


# ============================================================
# Step 1: Auto-label perturbation using energy spike
# ============================================================
def detect_perturbation_time(df, fs):
    """
    Detect perturbation timestamp using acceleration energy spike.
    Returns the time (in seconds) of the perturbation, or None for normal trials.
    """
    t = df['time_s'].values

    # Compute acc magnitude for all 3 sensors
    acc_energy = np.zeros(len(df))
    for sensor in ['thigh', 'shank', 'foot']:
        cols = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
        if all(c in df.columns for c in cols):
            mag = np.sqrt(df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2)
            acc_energy += (mag - 9.81)**2

    # Smooth with 500ms window
    window = int(0.5 * fs)
    if window < 1:
        window = 1
    energy_smooth = pd.Series(acc_energy).rolling(window, center=True).mean().values

    # Find the peak
    # Skip first and last 3 seconds (startup/shutdown artifacts)
    skip = int(3 * fs)
    if skip * 2 >= len(energy_smooth):
        skip = int(0.5 * fs)

    search_region = energy_smooth[skip:-skip] if skip > 0 else energy_smooth
    peak_idx = np.nanargmax(search_region) + skip

    peak_energy = energy_smooth[peak_idx]

    # Compute baseline energy (median of the signal)
    baseline = np.nanmedian(energy_smooth)

    # Peak must be at least 5x above baseline to count as perturbation
    if peak_energy > baseline * 5 and peak_energy > 30:
        return t[peak_idx], peak_energy, baseline
    else:
        return None, peak_energy, baseline


def auto_label_trial(filepath, is_trip=True):
    """Load a trial, detect perturbation, return labeled dataframe."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

    duration = df['time_s'].iloc[-1]
    fs = len(df) / duration

    name = Path(filepath).stem

    if is_trip:
        perturb_time, peak_e, baseline_e = detect_perturbation_time(df, fs)

        if perturb_time is not None:
            # Check if there was a manual label
            manual_events = df[df['perturbation_event'] == 1]
            manual_time = manual_events['time_s'].iloc[0] if len(manual_events) > 0 else None

            if manual_time is not None:
                diff = abs(perturb_time - manual_time)
                print(f"  {name}: auto={perturb_time:.1f}s, manual={manual_time:.1f}s, diff={diff:.1f}s, "
                      f"peak/baseline={peak_e/baseline_e:.1f}x")
            else:
                print(f"  {name}: auto={perturb_time:.1f}s (no manual label), "
                      f"peak/baseline={peak_e/baseline_e:.1f}x")

            return df, perturb_time, fs
        else:
            print(f"  {name}: WARNING - no clear perturbation detected! "
                  f"peak={peak_e:.1f}, baseline={baseline_e:.1f}")
            return df, None, fs
    else:
        print(f"  {name}: normal trial (no perturbation)")
        return df, None, fs


# ============================================================
# Step 2: Resample to uniform Hz
# ============================================================
def resample_trial(df, target_hz):
    """Resample a trial to a uniform sampling rate."""
    duration = df['time_s'].iloc[-1]
    n_samples = int(duration * target_hz)
    new_times = np.linspace(0, duration, n_samples)

    resampled = pd.DataFrame({'time_s': new_times})

    for col in FEATURE_COLS:
        if col in df.columns:
            valid = df[['time_s', col]].dropna()
            if len(valid) > 2:
                f = interp1d(valid['time_s'], valid[col], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
                resampled[col] = f(new_times)
            else:
                resampled[col] = 0.0
        else:
            resampled[col] = 0.0

    return resampled


# ============================================================
# Step 3: Sliding window segmentation
# ============================================================
def create_windows(df, perturb_time, target_hz, window_ms, stride_ms, horizons_ms):
    """
    Create sliding windows from a trial.

    For each window, create labels for multiple prediction horizons:
    - For each horizon H: label=1 if perturbation occurs within H ms after window ends

    Returns:
        windows: np.array of shape (n_windows, window_samples, n_features)
        labels: dict of {horizon_ms: np.array of shape (n_windows,)}
        window_times: list of (start_time, end_time) for each window
    """
    window_samples = int(window_ms / 1000 * target_hz)
    stride_samples = int(stride_ms / 1000 * target_hz)

    data = df[FEATURE_COLS].values
    times = df['time_s'].values
    n_samples = len(data)

    windows = []
    labels = {h: [] for h in horizons_ms}
    window_times = []

    i = 0
    while i + window_samples <= n_samples:
        window_data = data[i:i + window_samples]
        window_end_time = times[i + window_samples - 1]
        window_start_time = times[i]

        windows.append(window_data)
        window_times.append((window_start_time, window_end_time))

        # Label for each prediction horizon
        for h in horizons_ms:
            if perturb_time is not None:
                horizon_s = h / 1000.0
                # Label = 1 if perturbation happens within [0, horizon] seconds after window ends
                time_to_perturb = perturb_time - window_end_time
                if 0 <= time_to_perturb <= horizon_s:
                    labels[h].append(1)
                # Also label windows that overlap with the perturbation moment
                elif window_start_time <= perturb_time <= window_end_time:
                    labels[h].append(1)
                else:
                    labels[h].append(0)
            else:
                labels[h].append(0)

        i += stride_samples

    windows = np.array(windows, dtype=np.float32)
    for h in horizons_ms:
        labels[h] = np.array(labels[h], dtype=np.int64)

    return windows, labels, window_times


# ============================================================
# Step 4: Build dataset
# ============================================================
def build_dataset(data_dir, target_hz=100, window_ms=500, stride_ms=250,
                  horizons_ms=[100, 200, 300, 500]):
    """
    Process all trials and build the full dataset.
    """
    data_dir = Path(data_dir)
    files = sorted([f for f in data_dir.glob('*.csv') if not f.name.startswith('.')])

    # Skip slip_04 (test file)
    files = [f for f in files if f.name != 'slip_04.csv']

    print(f"\n{'='*60}")
    print(f"  BE124 Preprocessing Pipeline")
    print(f"  Files: {len(files)}")
    print(f"  Target Hz: {target_hz}")
    print(f"  Window: {window_ms}ms, Stride: {stride_ms}ms")
    print(f"  Horizons: {horizons_ms}ms")
    print(f"{'='*60}")

    all_windows = []
    all_labels = {h: [] for h in horizons_ms}
    all_subjects = []
    all_trial_types = []
    all_trial_names = []

    label_report = []

    # Process trips
    print(f"\n--- Trip Trials ---")
    trip_files = [f for f in files if f.name.startswith('trip_')]
    for filepath in trip_files:
        df, perturb_time, fs = auto_label_trial(str(filepath), is_trip=True)
        df_resampled = resample_trial(df, target_hz)

        windows, labels, wtimes = create_windows(
            df_resampled, perturb_time, target_hz, window_ms, stride_ms, horizons_ms)

        # Determine subject
        name = filepath.stem
        subject = 'hang' if 'hang' in name else 'xiaoyang'

        all_windows.append(windows)
        for h in horizons_ms:
            all_labels[h].append(labels[h])
        all_subjects.extend([subject] * len(windows))
        all_trial_types.extend(['trip'] * len(windows))
        all_trial_names.extend([name] * len(windows))

        n_pos = labels[horizons_ms[-1]].sum()
        label_report.append({
            'file': name, 'type': 'trip', 'subject': subject,
            'perturb_time': perturb_time, 'n_windows': len(windows),
            'n_positive': int(n_pos)
        })

    # Process normals
    print(f"\n--- Normal Trials ---")
    normal_files = [f for f in files if f.name.startswith('normal_')]
    for filepath in normal_files:
        df, perturb_time, fs = auto_label_trial(str(filepath), is_trip=False)
        df_resampled = resample_trial(df, target_hz)

        windows, labels, wtimes = create_windows(
            df_resampled, None, target_hz, window_ms, stride_ms, horizons_ms)

        name = filepath.stem
        subject = 'hang' if 'hang' in name else 'xiaoyang'

        all_windows.append(windows)
        for h in horizons_ms:
            all_labels[h].append(labels[h])
        all_subjects.extend([subject] * len(windows))
        all_trial_types.extend(['normal'] * len(windows))
        all_trial_names.extend([name] * len(windows))

        label_report.append({
            'file': name, 'type': 'normal', 'subject': subject,
            'perturb_time': None, 'n_windows': len(windows),
            'n_positive': 0
        })

    # Concatenate
    X = np.concatenate(all_windows, axis=0)
    Y = {}
    for h in horizons_ms:
        Y[h] = np.concatenate(all_labels[h], axis=0)
    subjects = np.array(all_subjects)
    trial_types = np.array(all_trial_types)
    trial_names = np.array(all_trial_names)

    return X, Y, subjects, trial_types, trial_names, label_report


# ============================================================
# Step 5: Train/Val/Test split
# ============================================================
def split_dataset(X, Y, subjects, trial_types, trial_names, horizons_ms, mode='random'):
    """
    Split dataset by trial (windows from same trial stay together).

    Modes:
      'random' (default): All trials mixed, 70% train / 15% val / 15% test
      'loso':  Leave-One-Subject-Out. Hang=train+val, Xiaoyang=test
    """
    splits = {}
    rng = np.random.RandomState(42)
    all_trials = np.unique(trial_names)

    if mode == 'loso':
        # LOSO: Xiaoyang = test, Hang = train + val
        test_trials = set([t for t in all_trials if 'xiaoyang' in t])
        hang_trials = np.array([t for t in all_trials if 'hang' in t])
        rng.shuffle(hang_trials)
        n_val = max(1, int(len(hang_trials) * 0.2))
        val_trials = set(hang_trials[:n_val])
        train_trials = set(hang_trials[n_val:])
    else:
        # Random: all trials mixed, 70/15/15 split
        rng.shuffle(all_trials)
        n = len(all_trials)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        test_trials = set(all_trials[:n_test])
        val_trials = set(all_trials[n_test:n_test + n_val])
        train_trials = set(all_trials[n_test + n_val:])

    train_mask = np.array([t in train_trials for t in trial_names])
    val_mask = np.array([t in val_trials for t in trial_names])
    test_mask = np.array([t in test_trials for t in trial_names])

    splits['train'] = {
        'X': X[train_mask],
        'subjects': subjects[train_mask],
        'trial_types': trial_types[train_mask],
        'trial_names': trial_names[train_mask],
    }
    splits['val'] = {
        'X': X[val_mask],
        'subjects': subjects[val_mask],
        'trial_types': trial_types[val_mask],
        'trial_names': trial_names[val_mask],
    }
    splits['test'] = {
        'X': X[test_mask],
        'subjects': subjects[test_mask],
        'trial_types': trial_types[test_mask],
        'trial_names': trial_names[test_mask],
    }

    for h in horizons_ms:
        splits['train'][f'Y_{h}'] = Y[h][train_mask]
        splits['val'][f'Y_{h}'] = Y[h][val_mask]
        splits['test'][f'Y_{h}'] = Y[h][test_mask]

    return splits, val_trials, train_trials, test_trials


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='BE124 Preprocessing Pipeline')
    parser.add_argument('--data-dir', default='data/', help='Raw data directory')
    parser.add_argument('--output-dir', default='dataset/', help='Output directory')
    parser.add_argument('--target-hz', type=int, default=100, help='Resample target Hz')
    parser.add_argument('--window-ms', type=int, default=500, help='Window size in ms')
    parser.add_argument('--stride-ms', type=int, default=250, help='Stride in ms')
    parser.add_argument('--horizons', nargs='+', type=int, default=[100, 200, 300, 500],
                        help='Prediction horizons in ms')
    parser.add_argument('--split', default='random', choices=['random', 'loso'],
                        help='Split mode: random (default, 70/15/15) or loso (leave-one-subject-out)')

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    X, Y, subjects, trial_types, trial_names, label_report = build_dataset(
        args.data_dir,
        target_hz=args.target_hz,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        horizons_ms=args.horizons
    )

    print(f"\n{'='*60}")
    print(f"  Dataset Built!")
    print(f"  X shape: {X.shape}  (windows, timesteps, features)")
    print(f"  Features: {N_FEATURES} ({', '.join(FEATURE_COLS[:6])}...)")
    for h in args.horizons:
        n_pos = Y[h].sum()
        print(f"  Y_{h}ms: {n_pos} positive / {len(Y[h])} total ({n_pos/len(Y[h])*100:.1f}%)")
    print(f"{'='*60}")

    # Split
    splits, val_trials, train_trials, test_trials = split_dataset(
        X, Y, subjects, trial_types, trial_names, args.horizons, mode=args.split)

    print(f"\n  Split mode: {args.split}")
    print(f"    Train: {len(splits['train']['X'])} windows ({len(train_trials)} trials)")
    print(f"    Val:   {len(splits['val']['X'])} windows ({len(val_trials)} trials)")
    print(f"    Test:  {len(splits['test']['X'])} windows ({len(test_trials)} trials)")

    for split_name in ['train', 'val', 'test']:
        for h in args.horizons:
            y = splits[split_name][f'Y_{h}']
            n_pos = y.sum()
            print(f"    {split_name} Y_{h}ms: {n_pos} pos / {len(y)} total ({n_pos/len(y)*100:.1f}%)")

    # Save
    print(f"\n  Saving to {out_dir}/...")

    for split_name in ['train', 'val', 'test']:
        np.save(out_dir / f'X_{split_name}.npy', splits[split_name]['X'])
        for h in args.horizons:
            np.save(out_dir / f'Y_{split_name}_{h}ms.npy', splits[split_name][f'Y_{h}'])

    # Save metadata
    metadata = {
        'target_hz': args.target_hz,
        'window_ms': args.window_ms,
        'stride_ms': args.stride_ms,
        'horizons_ms': args.horizons,
        'split_mode': args.split,
        'n_features': N_FEATURES,
        'feature_cols': FEATURE_COLS,
        'train_trials': sorted(list(train_trials)),
        'val_trials': sorted(list(val_trials)),
        'test_trials': sorted(list(test_trials)),
        'label_report': label_report,
        'shapes': {
            'X_train': list(splits['train']['X'].shape),
            'X_val': list(splits['val']['X'].shape),
            'X_test': list(splits['test']['X'].shape),
        }
    }

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  Files saved:")
    for f in sorted(out_dir.glob('*')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    {f.name}: {size_mb:.1f}MB")

    print(f"\n  Done! Ready for model training.")
    print(f"  Load with:")
    print(f"    X_train = np.load('{out_dir}/X_train.npy')")
    print(f"    Y_train = np.load('{out_dir}/Y_train_500ms.npy')")


if __name__ == '__main__':
    main()