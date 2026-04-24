#!/usr/bin/env python3
"""
BE124 Advanced Visualization
==============================
Additional plots beyond the basic visualize.py.
Focused on perturbation detection analysis.

Usage:
  python viz_advanced.py data/slip_hang_01.csv
  python viz_advanced.py data/ --batch --save

  # 单个trial全套分析
python viz_advanced.py data/slip_hang_01.csv --save


# 批量
python viz_advanced.py data/ --batch --save

# Slip vs Normal对比（最有用！）
python viz_advanced.py --compare data/slip_hang_01.csv data/normal_hang_01.csv --save

Requirements:
  pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram
from pathlib import Path
import argparse

COLORS = {
    'thigh': '#534AB7',
    'shank': '#1D9E75',
    'foot':  '#D85A30',
}


def load_and_prep(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Compute magnitudes
    for sensor in ['thigh', 'shank', 'foot']:
        acc = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
        gyro = [f'{sensor}_gyro_x', f'{sensor}_gyro_y', f'{sensor}_gyro_z']
        if all(c in df.columns for c in acc):
            df[f'{sensor}_acc_mag'] = np.sqrt(df[acc[0]]**2 + df[acc[1]]**2 + df[acc[2]]**2)
        if all(c in df.columns for c in gyro):
            df[f'{sensor}_gyro_mag'] = np.sqrt(df[gyro[0]]**2 + df[gyro[1]]**2 + df[gyro[2]]**2)

    # Jerk (derivative of acceleration)
    dt = df['time_s'].diff().median()
    for sensor in ['thigh', 'shank', 'foot']:
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_acc_{axis}'
            if col in df.columns:
                df[f'{sensor}_jerk_{axis}'] = df[col].diff() / dt
        if f'{sensor}_acc_mag' in df.columns:
            df[f'{sensor}_jerk_mag'] = df[f'{sensor}_acc_mag'].diff() / dt

    return df


# ============================================================
# 1. Jerk (rate of change of acceleration)
#    Perturbation shows as a sharp spike in jerk
# ============================================================
def plot_jerk(df, title="", save_path=None):
    """Plot jerk magnitude - key indicator for sudden perturbations."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Acceleration vs Jerk — {title}', fontsize=14, fontweight='bold')

    t = df['time_s']

    # Acc magnitude
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            axes[0].plot(t, df[col], linewidth=0.8, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].axhline(y=9.81, color='gray', linestyle=':', alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Jerk magnitude
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_jerk_mag'
        if col in df.columns:
            axes[1].plot(t, df[col].clip(-500, 500), linewidth=0.8,
                         color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
    axes[1].set_ylabel('Jerk (m/s³)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 2. Spectrogram (time-frequency view)
#    Shows how frequency content changes over time
# ============================================================
def plot_spectrogram(df, sensor='thigh', axis='acc_z', title="", save_path=None):
    """Time-frequency spectrogram to see frequency changes during perturbation."""
    col = f'{sensor}_{axis}'
    if col not in df.columns:
        print(f"  Column {col} not found")
        return

    data = df[col].dropna().values
    duration = df['time_s'].iloc[-1]
    fs = len(data) / duration

    fig, axes = plt.subplots(2, 1, figsize=(16, 7),
                              gridspec_kw={'height_ratios': [1, 2]}, sharex=True)
    fig.suptitle(f'Spectrogram — {sensor.upper()} {axis} — {title}',
                 fontsize=14, fontweight='bold')

    # Time domain on top
    t = np.linspace(0, duration, len(data))
    axes[0].plot(t, data, linewidth=0.5, color=COLORS[sensor])
    axes[0].set_ylabel(axis.split('_')[0])
    axes[0].grid(True, alpha=0.2)

    # Spectrogram on bottom
    nperseg = min(256, len(data) // 4)
    f, t_spec, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                       shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylim(0, min(30, fs/2))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 3. Cross-sensor correlation
#    How much do the 3 sensors move together?
# ============================================================
def plot_correlation(df, title="", save_path=None):
    """Rolling correlation between sensors - decorrelation may signal perturbation."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Inter-Sensor Correlation — {title}', fontsize=14, fontweight='bold')

    t = df['time_s']
    window = 100  # ~0.7s at 144Hz

    # Acc magnitude correlation
    pairs = [('thigh', 'shank'), ('thigh', 'foot'), ('shank', 'foot')]
    pair_colors = ['#534AB7', '#D85A30', '#1D9E75']

    for (s1, s2), color in zip(pairs, pair_colors):
        c1 = f'{s1}_acc_mag'
        c2 = f'{s2}_acc_mag'
        if c1 in df.columns and c2 in df.columns:
            corr = df[c1].rolling(window).corr(df[c2])
            axes[0].plot(t, corr, linewidth=0.8, color=color,
                         label=f'{s1.upper()}-{s2.upper()}', alpha=0.8)

    axes[0].set_ylabel('Acc correlation')
    axes[0].set_ylim(-1, 1)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Gyro magnitude correlation
    for (s1, s2), color in zip(pairs, pair_colors):
        c1 = f'{s1}_gyro_mag'
        c2 = f'{s2}_gyro_mag'
        if c1 in df.columns and c2 in df.columns:
            corr = df[c1].rolling(window).corr(df[c2])
            axes[1].plot(t, corr, linewidth=0.8, color=color,
                         label=f'{s1.upper()}-{s2.upper()}', alpha=0.8)

    axes[1].set_ylabel('Gyro correlation')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylim(-1, 1)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 4. Sliding window energy
#    Total signal energy in a moving window
# ============================================================
def plot_energy(df, window_ms=500, title="", save_path=None):
    """Sliding window signal energy - perturbations show as energy spikes."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f'Sliding Window Energy ({window_ms}ms) — {title}',
                 fontsize=14, fontweight='bold')

    t = df['time_s']
    duration = t.iloc[-1]
    fs = len(df) / duration
    window = int(window_ms / 1000 * fs)

    # Acc energy (sum of squared deviations from gravity)
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            energy = ((df[col] - 9.81) ** 2).rolling(window, center=True).mean()
            axes[0].plot(t, energy, linewidth=1, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)

    axes[0].set_ylabel('Acc energy (m²/s⁴)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Gyro energy
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns:
            energy = (df[col] ** 2).rolling(window, center=True).mean()
            axes[1].plot(t, energy, linewidth=1, color=COLORS[sensor],
                         label=sensor.upper(), alpha=0.8)

    axes[1].set_ylabel('Gyro energy (rad²/s²)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 5. 3D trajectory (acc space)
#    Visualize the acceleration vector movement in 3D
# ============================================================
def plot_3d_trajectory(df, sensor='foot', title="", save_path=None):
    """3D plot of acceleration vector - normal walking is a loop, perturbation breaks it."""
    cols = [f'{sensor}_acc_x', f'{sensor}_acc_y', f'{sensor}_acc_z']
    if not all(c in df.columns for c in cols):
        print(f"  Missing columns for {sensor}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample for clarity
    step = max(1, len(df) // 2000)
    x = df[cols[0]].values[::step]
    y = df[cols[1]].values[::step]
    z = df[cols[2]].values[::step]
    t = df['time_s'].values[::step]

    # Color by time
    scatter = ax.scatter(x, y, z, c=t, cmap='viridis', s=1, alpha=0.6)
    plt.colorbar(scatter, label='Time (s)', shrink=0.6)

    ax.set_xlabel(f'{sensor}_acc_x')
    ax.set_ylabel(f'{sensor}_acc_y')
    ax.set_zlabel(f'{sensor}_acc_z')
    ax.set_title(f'3D Acceleration Trajectory — {sensor.upper()} — {title}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 6. Sensor comparison dashboard
#    All key metrics in one view for quick assessment
# ============================================================
def plot_dashboard(df, title="", save_path=None):
    """Single-page dashboard with all key metrics."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Sensor Dashboard — {title}', fontsize=16, fontweight='bold')
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    t = df['time_s']
    duration = t.iloc[-1]
    fs = len(df) / duration

    # Row 1: Acc XYZ for each sensor
    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        ax = fig.add_subplot(gs[0, i])
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_acc_{axis}'
            if col in df.columns:
                ax.plot(t, df[col], linewidth=0.5, alpha=0.7, label=axis.upper())
        ax.set_title(f'{sensor.upper()} Acc', fontsize=11, color=COLORS[sensor])
        ax.set_ylabel('m/s²')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    # Row 2: Gyro XYZ for each sensor
    for i, sensor in enumerate(['thigh', 'shank', 'foot']):
        ax = fig.add_subplot(gs[1, i])
        for axis in ['x', 'y', 'z']:
            col = f'{sensor}_gyro_{axis}'
            if col in df.columns:
                ax.plot(t, df[col], linewidth=0.5, alpha=0.7, label=axis.upper())
        ax.set_title(f'{sensor.upper()} Gyro', fontsize=11, color=COLORS[sensor])
        ax.set_ylabel('rad/s')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    # Row 3: Signal magnitudes + energy
    ax = fig.add_subplot(gs[2, 0:2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_acc_mag'
        if col in df.columns:
            ax.plot(t, df[col], linewidth=0.8, color=COLORS[sensor],
                    label=sensor.upper(), alpha=0.8)
    ax.axhline(y=9.81, color='gray', linestyle=':', alpha=0.3)
    ax.set_title('Acceleration Magnitude', fontsize=11)
    ax.set_ylabel('m/s²')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 3 right: Gyro magnitude
    ax = fig.add_subplot(gs[2, 2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_gyro_mag'
        if col in df.columns:
            ax.plot(t, df[col], linewidth=0.8, color=COLORS[sensor],
                    label=sensor.upper(), alpha=0.8)
    ax.set_title('Gyro Magnitude', fontsize=11)
    ax.set_ylabel('rad/s')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 4: Jerk + Euler
    ax = fig.add_subplot(gs[3, 0:2])
    for sensor in ['thigh', 'shank', 'foot']:
        col = f'{sensor}_jerk_mag'
        if col in df.columns:
            ax.plot(t, df[col].clip(-500, 500), linewidth=0.6,
                    color=COLORS[sensor], label=sensor.upper(), alpha=0.7)
    ax.set_title('Jerk Magnitude (dAcc/dt)', fontsize=11)
    ax.set_ylabel('m/s³')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 4 right: Euler angles
    ax = fig.add_subplot(gs[3, 2])
    if 'foot_euler_y' in df.columns:
        ax.plot(t, df['foot_euler_y'], linewidth=1, color=COLORS['foot'],
                label='Roll', alpha=0.8)
    if 'foot_euler_z' in df.columns:
        ax.plot(t, df['foot_euler_z'], linewidth=1, color=COLORS['foot'],
                label='Pitch', alpha=0.6, linestyle='--')
    ax.set_title('Foot Euler Angles', fontsize=11)
    ax.set_ylabel('Degrees')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Add stats text
    stats_text = f'Duration: {duration:.1f}s | Hz: {fs:.0f} | Frames: {len(df)}'
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, color='gray')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# 7. Slip vs Normal comparison
# ============================================================
def compare_slip_normal(slip_path, normal_path, save_path=None):
    """Side-by-side comparison of a slip trial vs normal walking."""
    df_slip = load_and_prep(slip_path)
    df_norm = load_and_prep(normal_path)

    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharey='row')
    fig.suptitle(f'Slip vs Normal Comparison', fontsize=14, fontweight='bold')

    axes[0, 0].set_title(f'SLIP: {Path(slip_path).stem}', fontsize=11, color='red')
    axes[0, 1].set_title(f'NORMAL: {Path(normal_path).stem}', fontsize=11, color='green')

    for col_idx, (df, label) in enumerate([(df_slip, 'slip'), (df_norm, 'normal')]):
        t = df['time_s']

        # Acc magnitude
        for sensor in ['thigh', 'shank', 'foot']:
            col = f'{sensor}_acc_mag'
            if col in df.columns:
                axes[0, col_idx].plot(t, df[col], linewidth=0.8,
                                      color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
        axes[0, col_idx].axhline(y=9.81, color='gray', linestyle=':', alpha=0.3)
        axes[0, col_idx].set_ylabel('Acc mag (m/s²)')
        axes[0, col_idx].legend(fontsize=7)
        axes[0, col_idx].grid(True, alpha=0.2)

        # Gyro magnitude
        for sensor in ['thigh', 'shank', 'foot']:
            col = f'{sensor}_gyro_mag'
            if col in df.columns:
                axes[1, col_idx].plot(t, df[col], linewidth=0.8,
                                      color=COLORS[sensor], label=sensor.upper(), alpha=0.8)
        axes[1, col_idx].set_ylabel('Gyro mag (rad/s)')
        axes[1, col_idx].legend(fontsize=7)
        axes[1, col_idx].grid(True, alpha=0.2)

        # Jerk
        for sensor in ['thigh', 'shank', 'foot']:
            col = f'{sensor}_jerk_mag'
            if col in df.columns:
                axes[2, col_idx].plot(t, df[col].clip(-500, 500), linewidth=0.6,
                                      color=COLORS[sensor], label=sensor.upper(), alpha=0.7)
        axes[2, col_idx].set_ylabel('Jerk (m/s³)')
        axes[2, col_idx].set_xlabel('Time (s)')
        axes[2, col_idx].legend(fontsize=7)
        axes[2, col_idx].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================
def process_file(filepath, save=False):
    name = Path(filepath).stem
    print(f"\n  Processing: {name}")

    df = load_and_prep(filepath)

    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    prefix = fig_dir / name if save else None

    plot_dashboard(df, title=name,
                   save_path=f'{prefix}_dashboard.png' if save else None)
    plot_jerk(df, title=name,
              save_path=f'{prefix}_jerk.png' if save else None)
    plot_spectrogram(df, sensor='foot', axis='acc_z', title=name,
                     save_path=f'{prefix}_spectrogram.png' if save else None)
    plot_correlation(df, title=name,
                     save_path=f'{prefix}_correlation.png' if save else None)
    plot_energy(df, title=name,
                save_path=f'{prefix}_energy.png' if save else None)
    plot_3d_trajectory(df, sensor='foot', title=name,
                       save_path=f'{prefix}_3d.png' if save else None)


def main():
    parser = argparse.ArgumentParser(description='BE124 Advanced Visualization')
    parser.add_argument('paths', nargs='*', help='CSV file(s) or directory')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--batch', action='store_true', help='Process all CSVs')
    parser.add_argument('--compare', nargs=2, metavar=('SLIP', 'NORMAL'),
                        help='Compare slip vs normal trial')

    args = parser.parse_args()

    if args.compare:
        compare_slip_normal(args.compare[0], args.compare[1],
                            save_path='figures/slip_vs_normal.png' if args.save else None)
        return

    for p in args.paths:
        path = Path(p)
        if path.is_file():
            process_file(str(path), args.save)
        elif path.is_dir() and args.batch:
            for fp in sorted(path.glob('*.csv')):
                process_file(str(fp), args.save)


if __name__ == '__main__':
    main()