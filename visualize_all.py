#!/usr/bin/env python3
"""
BE124 IMU Visualization Pipeline (Merged)
==========================================
All visualizations in one script. Each trial gets its own folder.

Usage:
  # Single trial - all plots
  python visualize_all.py data/slip_hang_01.csv --save

  # Batch - all trials
  python visualize_all.py data/ --batch --save

  # Quick mode - only key plots (dashboard + magnitudes + jerk)
  python visualize_all.py data/slip_hang_01.csv --save --quick

  # Compare trip vs normal
  python visualize_all.py --compare data/slip_hang_01.csv data/normal_hang_01.csv --save

  # Overlay multiple trials
  python visualize_all.py data/slip_hang_01.csv data/slip_hang_02.csv --overlay

  # 单个trial全套11张图
python visualize_all.py data/slip_hang_01.csv --save

# 快速模式只出3张关键图（dashboard + magnitudes + jerk）
python visualize_all.py data/slip_hang_01.csv --save --quick

# 批量处理
python visualize_all.py data/ --batch --save

# Trip vs Normal对比
python visualize_all.py --compare data/slip_hang_01.csv data/normal_hang_01.csv --save

# 叠加多条trial
python visualize_all.py data/slip_hang_01.csv data/slip_hang_02.csv --overlay --save


Requirements:
  pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # non-interactive backend for batch processing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, spectrogram
from pathlib import Path
import argparse

# ============================================================
# Colors
# ============================================================
COLORS = {
    'thigh': '#534AB7',
    'shank': '#1D9E75',
    'foot':  '#D85A30',
}

# ============================================================
# Data loading & preprocessing
# ============================================================
def load_trial(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Magnitudes
    for sensor in ['thigh', 'shank', 'foot']:
        acc = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
        gyro = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']
        if all(c in df.columns for c in acc):
            df[f'{sensor}_acc_mag'] = np.sqrt(df[acc[0]]**2 + df[acc[1]]**2 + df[acc[2]]**2)
        if all(c in df.columns for c in gyro):
            df[f'{sensor}_gyro_mag'] = np.sqrt(df[gyro[0]]**2 + df[gyro[1]]**2 + df[gyro[2]]**2)

    # Jerk
    dt = df['time_s'].diff().median()
    if dt > 0:
        for sensor in ['thigh', 'shank', 'foot']:
            for axis in ['x', 'y', 'z']:
                col = f'{sensor}_acc_{axis}'
                if col in df.columns:
                    df[f'{sensor}_jerk_{axis}'] = df[col].diff() / dt
            if f'{sensor}_acc_mag' in df.columns:
                df[f'{sensor}_jerk_mag'] = df[f'{sensor}_acc_mag'].diff() / dt

    return df


def resample_to_hz(df, target_hz):
    duration = df['time_s'].iloc[-1]
    n_samples = int(duration * target_hz)
    new_times = np.linspace(0, duration, n_samples)
    resampled = pd.DataFrame({'time_s': new_times})
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['timestamp', 'time_s']]
    for col in numeric_cols:
        valid = df[['time_s', col]].dropna()
        if len(valid) > 2:
            f = interp1d(valid['time_s'], valid[col], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            resampled[col] = f(new_times)
    return resampled


def print_summary(df, filepath):
    duration = df['time_s'].iloc[-1]
    n = len(df)
    hz = n / duration if duration > 0 else 0
    events = int(df['perturbation_event'].sum()) if 'perturbation_event' in df.columns else 0
    diffs = df['time_s'].diff().dropna()
    max_gap = diffs.max() * 1000
    big_gaps = (diffs > 0.02).sum()

    print(f"\n{'='*60}")
    print(f"  {Path(filepath).stem}")
    print(f"  {n:,} frames | {duration:.1f}s | {hz:.0f}Hz")
    print(f"  Max gap: {max_gap:.0f}ms | Gaps>20ms: {big_gaps}")
    print(f"  Events: {events}")
    print(f"{'='*60}")


# ============================================================
# Plot functions
# ============================================================
def _mark_events(df, axes):
    """Add red vertical lines at perturbation events."""
    if 'perturbation_event' not in df.columns:
        return
    events = df[df['perturbation_event'] == 1]
    ax_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _, row in events.iterrows():
        for ax in ax_list:
            ax.axvline(x=row['time_s'], color='red', alpha=0.5,
                       linewidth=1, linestyle='--')


def plot_overview(df, title="", save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(f'IMU Overview — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']

    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        color = COLORS[sensor]
        for j, axis in enumerate(['x', 'y', 'z']):
            col = f'{sensor}_acc_{axis}'
            if col in df.columns:
                axes[i, 0].plot(t, df[col], lw=0.8, alpha=[1, 0.6, 0.4][j],
                                label=axis.upper(), color=color)
            col = f'{sensor}_gyro_{axis}'
            if col in df.columns:
                axes[i, 1].plot(t, df[col], lw=0.8, alpha=[1, 0.6, 0.4][j],
                                label=axis.upper(), color=color)
        axes[i, 0].set_ylabel('m/s²')
        axes[i, 0].set_title(f'{sensor.upper()} Acc', fontsize=11, color=color)
        axes[i, 0].legend(fontsize=7, loc='upper right')
        axes[i, 0].grid(True, alpha=0.2)
        axes[i, 1].set_ylabel('rad/s')
        axes[i, 1].set_title(f'{sensor.upper()} Gyro', fontsize=11, color=color)
        axes[i, 1].legend(fontsize=7, loc='upper right')
        axes[i, 1].grid(True, alpha=0.2)

    _mark_events(df, axes)
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_magnitudes(df, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Signal Magnitudes — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            axes[0].plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[0].axhline(y=9.81, color='gray', ls=':', alpha=0.3, label='Gravity')
    axes[0].set_ylabel('Acc magnitude (m/s²)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns:
            axes[1].plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[1].set_ylabel('Gyro magnitude (rad/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    _mark_events(df, axes)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_jerk(df, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Acceleration vs Jerk — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            axes[0].plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[0].axhline(y=9.81, color='gray', ls=':', alpha=0.3)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_jerk_mag'
        if col in df.columns:
            axes[1].plot(t, df[col].clip(-500, 500), lw=0.8, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)
    axes[1].set_ylabel('Jerk (m/s³)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    _mark_events(df, axes)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_joints(df, title="", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    fig.suptitle(f'Foot Euler Angles — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']
    if 'foot_euler_y' in df.columns:
        ax.plot(t, df['foot_euler_y'], lw=1, color=COLORS['foot'], label='Roll (Y)', alpha=0.8)
    if 'foot_euler_z' in df.columns:
        ax.plot(t, df['foot_euler_z'], lw=1, color=COLORS['foot'], label='Pitch (Z)',
                alpha=0.6, ls='--')
    ax.set_ylabel('Degrees')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    _mark_events(df, ax)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_fft(df, sensor='thigh', axis='acc_z', title="", save_path=None):
    col = f'{sensor}_{axis}'
    if col not in df.columns: return
    data = df[col].dropna().values
    if len(data) < 64: return
    duration = df['time_s'].iloc[-1]
    fs = len(data) / duration
    fft_vals = np.abs(np.fft.rfft(data - np.mean(data)))
    freqs = np.fft.rfftfreq(len(data), d=1/fs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(freqs, fft_vals, lw=0.8, color=COLORS[sensor])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'FFT — {sensor.upper()} {axis} — {title}', fontsize=12)
    ax.set_xlim(0, fs/2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_spectrogram(df, sensor='foot', axis='acc_z', title="", save_path=None):
    col = f'{sensor}_{axis}'
    if col not in df.columns: return
    data = df[col].dropna().values
    duration = df['time_s'].iloc[-1]
    fs = len(data) / duration

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)
    fig.suptitle(f'Spectrogram — {sensor.upper()} {axis} — {title}', fontsize=14, fontweight='bold')
    t = np.linspace(0, duration, len(data))
    axes[0].plot(t, data, lw=0.5, color=COLORS[sensor])
    axes[0].set_ylabel(axis)
    axes[0].grid(True, alpha=0.2)

    nperseg = min(256, len(data) // 4)
    f, t_s, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    axes[1].pcolormesh(t_s, f, 10*np.log10(Sxx+1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylim(0, min(30, fs/2))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation(df, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Inter-Sensor Correlation — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']
    duration = t.iloc[-1]
    fs = len(df) / duration
    window = int(0.7 * fs)  # 0.7s window

    pairs = [('thigh', 'shank'), ('thigh', 'foot'), ('shank', 'foot')]
    pair_colors = ['#534AB7', '#D85A30', '#1D9E75']

    for metric_idx, metric in enumerate(['acc_mag', 'gyro_mag']):
        for (s1, s2), color in zip(pairs, pair_colors):
            c1, c2 = f'{s1}_{metric}', f'{s2}_{metric}'
            if c1 in df.columns and c2 in df.columns:
                corr = df[c1].rolling(window).corr(df[c2])
                axes[metric_idx].plot(t, corr, lw=0.8, color=color,
                                      label=f'{s1.upper()}-{s2.upper()}', alpha=0.8)
        label = 'Acc' if metric_idx == 0 else 'Gyro'
        axes[metric_idx].set_ylabel(f'{label} correlation')
        axes[metric_idx].set_ylim(-1, 1)
        axes[metric_idx].axhline(y=0, color='gray', ls=':', alpha=0.3)
        axes[metric_idx].legend(fontsize=9)
        axes[metric_idx].grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_energy(df, window_ms=500, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Signal Energy ({window_ms}ms window) — {title}', fontsize=14, fontweight='bold')
    t = df['time_s']
    duration = t.iloc[-1]
    fs = len(df) / duration
    window = int(window_ms / 1000 * fs)

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            energy = ((df[col] - 9.81)**2).rolling(window, center=True).mean()
            axes[0].plot(t, energy, lw=1, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[0].set_ylabel('Acc energy (m²/s⁴)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns:
            energy = (df[col]**2).rolling(window, center=True).mean()
            axes[1].plot(t, energy, lw=1, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[1].set_ylabel('Gyro energy (rad²/s²)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    _mark_events(df, axes)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d(df, sensor='foot', title="", save_path=None):
    cols = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
    if not all(c in df.columns for c in cols): return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(df) // 2000)
    x, y, z = df[cols[0]].values[::step], df[cols[1]].values[::step], df[cols[2]].values[::step]
    t = df['time_s'].values[::step]
    sc = ax.scatter(x, y, z, c=t, cmap='viridis', s=1, alpha=0.6)
    plt.colorbar(sc, label='Time (s)', shrink=0.6)
    ax.set_xlabel('acc_x'); ax.set_ylabel('acc_y'); ax.set_zlabel('acc_z')
    ax.set_title(f'3D Acc Trajectory — {sensor.upper()} — {title}')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dashboard(df, title="", save_path=None):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Dashboard — {title}', fontsize=16, fontweight='bold')
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    t = df['time_s']
    duration = t.iloc[-1]
    fs = len(df) / duration

    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        ax = fig.add_subplot(gs[0, i])
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_acc_{axis}'
            if col in df.columns: ax.plot(t, df[col], lw=0.5, alpha=0.7, label=axis.upper())
        ax.set_title(f'{sensor.upper()} Acc', fontsize=11, color=COLORS[sensor])
        ax.set_ylabel('m/s²'); ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        ax = fig.add_subplot(gs[1, i])
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_gyro_{axis}'
            if col in df.columns: ax.plot(t, df[col], lw=0.5, alpha=0.7, label=axis.upper())
        ax.set_title(f'{sensor.upper()} Gyro', fontsize=11, color=COLORS[sensor])
        ax.set_ylabel('rad/s'); ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, 0:2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns: ax.plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    ax.axhline(y=9.81, color='gray', ls=':', alpha=0.3)
    ax.set_title('Acc Magnitude'); ax.set_ylabel('m/s²'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, 2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns: ax.plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    ax.set_title('Gyro Magnitude'); ax.set_ylabel('rad/s'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[3, 0:2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_jerk_mag'
        if col in df.columns: ax.plot(t, df[col].clip(-500, 500), lw=0.6, color=COLORS[sensor], label=sensor.upper(), alpha=0.7)
    ax.set_title('Jerk (dAcc/dt)'); ax.set_ylabel('m/s³'); ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[3, 2])
    if 'foot_euler_y' in df.columns: ax.plot(t, df['foot_euler_y'], lw=1, color=COLORS['foot'], label='Roll', alpha=0.8)
    if 'foot_euler_z' in df.columns: ax.plot(t, df['foot_euler_z'], lw=1, color=COLORS['foot'], label='Pitch', alpha=0.6, ls='--')
    ax.set_title('Foot Euler'); ax.set_ylabel('Degrees'); ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.text(0.5, 0.01, f'Duration: {duration:.1f}s | Hz: {fs:.0f} | Frames: {len(df)}',
             ha='center', fontsize=10, color='gray')
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_downsample(df, title="", save_path=None):
    duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
    original_hz = int(len(df) / duration)
    rates = [original_hz, 100, 80, 50]
    fig, axes = plt.subplots(len(rates), 1, figsize=(16, 3*len(rates)), sharex=True)
    fig.suptitle(f'Downsampling — Thigh Acc Mag — {title}', fontsize=14, fontweight='bold')

    for i, hz in enumerate(rates):
        r = resample_to_hz(df, hz)
        for s in ['thigh']:
            acc = [f'{s}_acc_x', f'{s}_acc_y', f'{s}_acc_z']
            if all(c in r.columns for c in acc):
                r[f'{s}_acc_mag'] = np.sqrt(r[acc[0]]**2 + r[acc[1]]**2 + r[acc[2]]**2)
        if 'thigh_acc_mag' in r.columns:
            axes[i].plot(r['time_s'], r['thigh_acc_mag'], lw=0.8, color=COLORS['thigh'])
        axes[i].set_ylabel('m/s²')
        axes[i].set_title(f'{hz} Hz ({len(r)} samples)', fontsize=11)
        axes[i].axhline(y=9.81, color='gray', ls=':', alpha=0.3)
        axes[i].grid(True, alpha=0.2)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_trials(slip_path, normal_path, save_path=None):
    df_s = load_trial(slip_path)
    df_n = load_trial(normal_path)

    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharey='row')
    fig.suptitle('Trip vs Normal Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_title(f'TRIP: {Path(slip_path).stem}', fontsize=11, color='red')
    axes[0, 1].set_title(f'NORMAL: {Path(normal_path).stem}', fontsize=11, color='green')

    for ci, df in enumerate([df_s, df_n]):
        t = df['time_s']
        for sensor in ['thigh', 'shank', 'foot']:
            col = f'{sensor}_acc_mag'
            if col in df.columns:
                axes[0, ci].plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
            col = f'{sensor}_gyro_mag'
            if col in df.columns:
                axes[1, ci].plot(t, df[col], lw=0.8, color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
            col = f'{sensor}_jerk_mag'
            if col in df.columns:
                axes[2, ci].plot(t, df[col].clip(-500, 500), lw=0.6, color=COLORS[sensor], label=sensor.upper(), alpha=0.7)

        axes[0, ci].axhline(y=9.81, color='gray', ls=':', alpha=0.3)
        for r in range(3):
            axes[r, ci].legend(fontsize=7); axes[r, ci].grid(True, alpha=0.2)

    axes[0, 0].set_ylabel('Acc mag (m/s²)')
    axes[1, 0].set_ylabel('Gyro mag (rad/s)')
    axes[2, 0].set_ylabel('Jerk (m/s³)')
    axes[2, 0].set_xlabel('Time (s)'); axes[2, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def overlay_trials(filepaths, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle('Trial Overlay', fontsize=14, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(filepaths)))

    for i, fp in enumerate(filepaths):
        df = load_trial(fp)
        t = df['time_s']
        label = Path(fp).stem
        if 'thigh_acc_mag' in df.columns:
            axes[0].plot(t, df['thigh_acc_mag'], lw=0.8, color=colors[i], label=label, alpha=0.7)
        if 'thigh_gyro_mag' in df.columns:
            axes[1].plot(t, df['thigh_gyro_mag'], lw=0.8, color=colors[i], label=label, alpha=0.7)

    axes[0].set_ylabel('Thigh Acc Mag (m/s²)'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.2)
    axes[1].set_ylabel('Thigh Gyro Mag (rad/s)'); axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Main pipeline
# ============================================================
def process_trial(filepath, save=False, quick=False):
    name = Path(filepath).stem
    df = load_trial(filepath)
    print_summary(df, filepath)

    if save:
        out_dir = Path('figures') / name
        out_dir.mkdir(parents=True, exist_ok=True)
        p = lambda fname: str(out_dir / fname)
    else:
        p = lambda fname: None

    # Always generate these
    plot_dashboard(df, name, p('dashboard.png'))
    plot_magnitudes(df, name, p('magnitudes.png'))
    plot_jerk(df, name, p('jerk.png'))

    if not quick:
        plot_overview(df, name, p('overview.png'))
        plot_joints(df, name, p('joints.png'))
        plot_fft(df, 'thigh', 'acc_z', name, p('fft_thigh.png'))
        plot_spectrogram(df, 'foot', 'acc_z', name, p('spectrogram.png'))
        plot_correlation(df, name, p('correlation.png'))
        plot_energy(df, 500, name, p('energy.png'))
        plot_3d(df, 'foot', name, p('3d.png'))
        plot_downsample(df, name, p('downsample.png'))

    if save:
        print(f"  Saved {3 if quick else 11} plots to figures/{name}/")


def main():
    parser = argparse.ArgumentParser(description='BE124 Visualization Pipeline')
    parser.add_argument('paths', nargs='*', help='CSV file(s) or directory')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--batch', action='store_true', help='Process all CSVs in directory')
    parser.add_argument('--quick', action='store_true', help='Only key plots (dashboard + magnitudes + jerk)')
    parser.add_argument('--compare', nargs=2, metavar=('TRIP', 'NORMAL'), help='Compare trip vs normal')
    parser.add_argument('--overlay', action='store_true', help='Overlay multiple trials')

    args = parser.parse_args()

    if args.compare:
        out = 'figures/trip_vs_normal.png' if args.save else None
        compare_trials(args.compare[0], args.compare[1], out)
        if out: print(f"  Saved: {out}")
        return

    if args.overlay and args.paths:
        out = 'figures/overlay.png' if args.save else None
        overlay_trials(args.paths, out)
        if out: print(f"  Saved: {out}")
        return

    for p in args.paths:
        path = Path(p)
        if path.is_file():
            process_trial(str(path), args.save, args.quick)
        elif path.is_dir():
            csvs = sorted(path.glob('*.csv'))
            print(f"\n  Found {len(csvs)} files in {path}")
            for fp in csvs:
                process_trial(str(fp), args.save, args.quick)


if __name__ == '__main__':
    main()