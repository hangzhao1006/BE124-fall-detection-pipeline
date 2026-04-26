# BE124 Data Processing Pipeline
# 数据处理流程文档

---

## Overview / 总览

```
Raw CSV (60 trials, ~160Hz)
    │
    ▼
[1] Auto-label perturbation timestamps (energy spike detection)
    自动标注绊倒时间点（能量尖峰检测）
    │
    ▼
[2] Resample to uniform Hz (50/80/100/120/160)
    统一采样率
    │
    ▼
[3] Sliding window segmentation (500ms window, 250ms stride)
    滑动窗口切片
    │
    ▼
[4] Label windows for multiple prediction horizons (100/200/300/500ms)
    按不同预测时间窗口标注
    │
    ▼
[5] Train/Val/Test split (Leave-One-Subject-Out)
    数据集划分
    │
    ▼
NumPy arrays ready for model training
准备好的训练数据
```

---

## Step 1: Auto-Labeling / 自动标注

### Problem / 问题
Most trip trials don't have manual perturbation labels (only 11 out of 40 had spacebar pressed during collection). We need to find the exact perturbation time for each trial.

大部分trip trial没有手动标注（40条中只有11条按了空格）。需要自动找到每条trial的perturbation时间。

### Method / 方法
1. Compute acceleration magnitude for all 3 sensors: $|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}$
2. Compute energy: $E = (|a_{thigh}| - g)^2 + (|a_{shank}| - g)^2 + (|a_{foot}| - g)^2$
3. Smooth with 500ms rolling window
4. Find the peak energy (skip first/last 3 seconds to avoid artifacts)
5. If peak > 5× median energy AND peak > 30: mark as perturbation

### Validation / 验证
For trials with manual labels, auto vs manual comparison:

| Trial | Auto (s) | Manual (s) | Diff (s) |
|-------|----------|-----------|----------|
| trip_xiaoyang_01 | 23.7 | 23.8 | 0.2 |
| trip_xiaoyang_02 | 20.9 | 21.2 | 0.3 |
| trip_xiaoyang_03 | 20.1 | 21.7 | 1.5 |
| trip_xiaoyang_04 | 21.0 | 22.9 | 1.9 |
| trip_xiaoyang_05 | 36.6 | 37.3 | 0.7 |
| trip_hang_01 | 21.6 | 22.1 | 0.5 |
| trip_hang_04 | 30.1 | 33.1 | 3.1 |

Xiaoyang's trials: average diff = 0.9s (very accurate)
Hang's some trials had larger diff due to inaccurate manual labeling

### Peak/Baseline Ratio / 峰值基线比
All trip trials have strong perturbation signals:
- Minimum ratio: 6.3x (trip_hang_02)
- Maximum ratio: 5189x (trip_xiaoyang_13)
- Most trials: 20-500x above baseline

This means the perturbation event is always clearly detectable in the IMU data.

---

## Step 2: Resampling / 统一采样率

### Why / 为什么
Raw data Hz varies between trials (155-164Hz). Models need uniform input size. We also want to compare model performance at different sampling rates.

原始数据采样率不完全一致（155-164Hz）。模型需要统一的输入大小。我们还要对比不同采样率对模型性能的影响。

### Method / 方法
Linear interpolation to target Hz. Creates evenly-spaced time points and interpolates all sensor values.

线性插值到目标Hz。创建等间距时间点，插值所有传感器数据。

### Generated Datasets / 生成的数据集

| Dataset | Target Hz | Timesteps per window | X shape | Size |
|---------|-----------|---------------------|---------|------|
| dataset_50hz | 50 | 25 | (17597, 25, 18) | ~30MB |
| dataset_80hz | 80 | 40 | (16891, 40, 18) | ~47MB |
| dataset_100hz | 100 | 50 | (16891, 50, 18) | ~58MB |
| dataset_120hz | 120 | 60 | (16891, 60, 18) | ~70MB |
| dataset_160hz | 160 | 80 | (16891, 80, 18) | ~93MB |

Timesteps = Hz × window_duration = Hz × 0.5s

---

## Step 3: Sliding Window / 滑动窗口

### Parameters / 参数
- **Window size**: 500ms (captures approximately half a gait cycle)
- **Stride**: 250ms (50% overlap between consecutive windows)
- **Features per window**: 18 channels

### Why 500ms? / 为什么500ms？
- Normal walking gait cycle ≈ 1 second, so 500ms ≈ half a cycle
- Pre-fall biomechanical changes occur within 200-500ms before impact
- Literature on IMU fall detection commonly uses 200-500ms windows
- Can be tuned later (250ms, 750ms, 1000ms) as ablation experiment

### Why 250ms stride? / 为什么250ms步长？
- 50% overlap ensures no event is missed between windows
- Creates more training samples (important with limited data)
- Standard practice in time-series classification

### Features / 特征（18维）

```
Input tensor shape: (n_windows, timesteps, 18)

Channels:
 0: thigh_acc_x       6: shank_acc_x      12: foot_acc_x
 1: thigh_acc_y       7: shank_acc_y      13: foot_acc_y
 2: thigh_acc_z       8: shank_acc_z      14: foot_acc_z
 3: thigh_gyro_x      9: shank_gyro_x     15: foot_gyro_x
 4: thigh_gyro_y     10: shank_gyro_y     16: foot_gyro_y
 5: thigh_gyro_z     11: shank_gyro_z     17: foot_gyro_z
```

**NOT included** (and why):
- Magnetometer (noisy indoors, not relevant for fall prediction)
- Euler angles (only available for foot, would create asymmetry)
- Derived features like jerk, energy (let the model learn these)

---

## Step 4: Labeling Windows / 窗口标注

### Task / 任务
Binary classification: will perturbation happen soon?
二分类：perturbation是否即将发生？

### Multiple Prediction Horizons / 多个预测时间窗口

For each window, we create labels for 4 different prediction horizons:

```
Window ends here
      │
      ▼
──────┤
      │◄── 100ms ──►│  Y_100ms: perturbation within 100ms?
      │◄──── 200ms ────►│  Y_200ms: perturbation within 200ms?
      │◄────── 300ms ──────►│  Y_300ms: perturbation within 300ms?
      │◄────────── 500ms ──────────►│  Y_500ms: perturbation within 500ms?
```

A window is labeled **positive (1)** if:
- Perturbation occurs within H ms after the window ends, OR
- Perturbation occurs during the window itself

A window is labeled **negative (0)** otherwise.

### Label Distribution / 标签分布 (100Hz dataset)

| Horizon | Positive | Negative | Positive % |
|---------|----------|----------|-----------|
| 100ms | 101 | 16790 | 0.6% |
| 200ms | 110 | 16781 | 0.7% |
| 300ms | 129 | 16762 | 0.8% |
| 500ms | 156 | 16735 | 0.9% |

**Class imbalance is severe (~99:1)** — this is expected because perturbation is a rare event within each 70s trial. Will be handled in model training with class weights or focal loss.

### Why these horizons? / 为什么选这些时间窗口？
- **100ms**: Very short warning time. Can the model detect perturbation onset?
- **200ms**: Minimal useful reaction time for a human
- **300ms**: Moderate prediction window
- **500ms**: Half-second advance warning — most useful for real-time intervention (e.g., triggering an actuator)

This enables the **accuracy vs. prediction horizon** analysis that Tara (TA) suggested as the key result.

---

把 DATA_PROCESSING.md 里 `## Step 5` 那一整段替换成这个：

---

## Step 5: Train/Val/Test Split / 数据集划分

### Default: Random Split (recommended) / 默认：随机划分

All trials from both subjects mixed together, split by trial (windows from same trial stay together):

两个被试的数据混合在一起，按trial划分（同一条trial的window不会跨split）：

- **Train**: 70% of trials → ~11,827 windows
- **Val**: 15% of trials → ~2,479 windows
- **Test**: 15% of trials → ~2,585 windows

Both Hang and Xiaoyang's data appear in all three splits. This gives the model more training data and learns from both subjects' gait patterns.

两个人的数据都出现在三个split里。训练数据更多，模型能学到两个人的步态模式。

### Optional: LOSO (Leave-One-Subject-Out) / 可选：跨被试验证

For evaluating cross-subject generalization:

- **Train + Val**: Hang's data only (80/20 split)
- **Test**: Xiaoyang's data only

This tests if the model can generalize to a completely unseen person. Expect lower accuracy than random split, but it's a more rigorous evaluation.

用于评估跨被试泛化能力。准确率会比random split低，但评估更严格。

### Usage / 用法

```bash
# Default random split
python preprocess.py --data-dir data/ --output-dir dataset_100hz/

# LOSO split (for additional experiment)
python preprocess.py --data-dir data/ --output-dir dataset_100hz_loso/ --split loso
```

### Comparison / 对比

| | Random | LOSO |
|---|---|---|
| Train windows | 11,827 | 6,636 |
| Val windows | 2,479 | 1,627 |
| Test windows | 2,585 | 8,628 |
| Train subjects | Both | Hang only |
| Test subjects | Both | Xiaoyang only |
| Use case | Primary results | Generalization experiment |

---

## How to Run / 如何运行

```bash
# Generate dataset at 100Hz (recommended to start)
python preprocess.py --data-dir data/ --output-dir dataset_100hz/

# Generate at other frequencies for ablation
python preprocess.py --data-dir data/ --output-dir dataset/dataset_50hz/ --target-hz 50
python preprocess.py --data-dir data/ --output-dir dataset/dataset_80hz/ --target-hz 80
python preprocess.py --data-dir data/ --output-dir dataset/dataset_120hz/ --target-hz 120
python preprocess.py --data-dir data/ --output-dir dataset/dataset_160hz/ --target-hz 160

# Custom window size (e.g., 750ms window)
python preprocess.py --data-dir data/ --output-dir dataset_100hz_750ms/ --window-ms 750

# Custom stride
python preprocess.py --data-dir data/ --output-dir dataset_100hz_s125/ --stride-ms 125
```

---

## Output Files / 输出文件

```
dataset_100hz/
├── X_train.npy          # (6636, 50, 18) float32 — training windows
├── X_val.npy            # (1627, 50, 18) float32 — validation windows
├── X_test.npy           # (8628, 50, 18) float32 — test windows
├── Y_train_100ms.npy    # (6636,) int64 — labels for 100ms horizon
├── Y_train_200ms.npy    # (6636,) int64
├── Y_train_300ms.npy    # (6636,) int64
├── Y_train_500ms.npy    # (6636,) int64
├── Y_val_100ms.npy      # (1627,) int64
├── Y_val_200ms.npy      # ...
├── Y_val_300ms.npy
├── Y_val_500ms.npy
├── Y_test_100ms.npy     # (8628,) int64
├── Y_test_200ms.npy
├── Y_test_300ms.npy
├── Y_test_500ms.npy
└── metadata.json        # Config, feature names, split info, label report
```

### Loading in Python / 加载方式

```python
import numpy as np

# Load 100Hz dataset with 500ms prediction horizon
X_train = np.load('dataset_100hz/X_train.npy')  # (6636, 50, 18)
Y_train = np.load('dataset_100hz/Y_train_500ms.npy')  # (6636,)
X_val = np.load('dataset_100hz/X_val.npy')
Y_val = np.load('dataset_100hz/Y_val_500ms.npy')
X_test = np.load('dataset_100hz/X_test.npy')
Y_test = np.load('dataset_100hz/Y_test_500ms.npy')

print(f"Train: {X_train.shape}, pos={Y_train.sum()}/{len(Y_train)}")
print(f"Val:   {X_val.shape}, pos={Y_val.sum()}/{len(Y_val)}")
print(f"Test:  {X_test.shape}, pos={Y_test.sum()}/{len(Y_test)}")
```

---

## Next Steps / 下一步

1. **Model training**: Start with GRU baseline on dataset_100hz
2. **Ablation experiments**:
   - Sampling rate: compare 50/80/100/120/160Hz
   - Window size: compare 250/500/750/1000ms
   - Prediction horizon: compare 100/200/300/500ms
3. **Class imbalance handling**: class weights, focal loss, or oversampling
4. **Feature importance**: SHAP analysis to identify most important sensors/axes
