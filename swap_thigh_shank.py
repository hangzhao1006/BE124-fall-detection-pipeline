#!/usr/bin/env python3
"""
Swap THIGH and SHANK columns in all CSV files.
Fixes the CH_THIGH=5/CH_SHANK=4 wiring mistake.

Usage:
  python swap_thigh_shank.py data/
  python swap_thigh_shank.py data/slip_hang_01.csv
"""

import pandas as pd
from pathlib import Path
import sys

def swap_file(filepath):
    df = pd.read_csv(filepath)
    
    # Build rename mapping
    rename = {}
    for col in df.columns:
        if col.startswith('thigh_'):
            rename[col] = col.replace('thigh_', 'shank_')
        elif col.startswith('shank_'):
            rename[col] = col.replace('shank_', 'thigh_')
    
    df = df.rename(columns=rename)
    
    # Reorder columns to keep thigh before shank
    original_order = pd.read_csv(filepath, nrows=0).columns.tolist()
    df = df[original_order]
    
    df.to_csv(filepath, index=False)
    print(f"  Swapped: {filepath}")

# Process args
for p in sys.argv[1:]:
    path = Path(p)
    if path.is_file() and path.suffix == '.csv':
        swap_file(str(path))
    elif path.is_dir():
        csvs = sorted(path.glob('*.csv'))
        print(f"  Found {len(csvs)} files in {path}")
        for f in csvs:
            swap_file(str(f))
        print(f"  Done! {len(csvs)} files fixed.")