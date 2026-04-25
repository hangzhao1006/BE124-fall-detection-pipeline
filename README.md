# BE124 Fall Perturbation Detection — Experiment & Analysis Handbook
# BE124 人工反射弧项目 — 实验采集与分析手册

---

## 一、Project Overview / 实验目标

Use 3 IMUs (thigh, shank, foot) on the dominant leg to collect gait data, then train ML models to predict perturbation (pre-fall events).

用3个IMU（大腿、小腿、脚）采集步态数据，训练模型预测perturbation（摔倒前兆）。

---

## 二、Hardware / 硬件清单

| Item 物品 | Qty 数量 | Notes 说明 |
|-----------|---------|-----------|
| Adafruit ESP32-S3 Feather (2MB PSRAM) | 1 | Main controller 主控板 |
| PCA9548A I2C Multiplexer | 1 | Routes I2C to 3 sensors / I2C多路复用 |
| Adafruit ICM-20948 9-DoF IMU | 2 | Thigh + Shank / 大腿 + 小腿 |
| Adafruit BNO055 9-DoF IMU | 1 | Foot / 脚 |
| Straps / Velcro 绑带 | 3 sets | Secure IMUs to leg / 固定IMU |
| USB Power Bank 充电宝 | 1 | Powers ESP32 / 给ESP32供电 |
| Waist pouch 腰包 | 1 | Holds ESP32 + mux + power / 装硬件 |
| Phone 手机 | 1 | Hotspot + video / 热点 + 录视频 |
| Laptop 笔记本 | 1 | Runs udp_logger.py / 采集脚本 |
| Obstacle (block/tape) 障碍物 | 1-2 | Trip scenario / 绊倒场景 |

---

## 三、IMU Placement / 传感器安装

All mounted on **dominant leg（惯用腿）**:

```
         ┌──────────┐
         │  THIGH   │ ← ICM-20948 #1, outer mid-thigh / 大腿外侧中段
         │  大腿    │   PCA9548A Channel 4
         └────┬─────┘
              │
         ┌────┴─────┐
         │  SHANK   │ ← ICM-20948 #2, outer mid-shin / 小腿胫骨外侧中段
         │  小腿    │   PCA9548A Channel 5
         └────┬─────┘
              │
         ┌────┴─────┐
         │   FOOT   │ ← BNO055, top of shoe / 鞋面脚背
         │    脚    │   PCA9548A Channel 7
         └──────────┘
```

Key points / 安装要点:
- Keep XYZ axes consistent across all 3 IMUs / IMU朝向尽量一致
- Strap tightly — loose sensors add noise / 绑紧防止晃动
- Place a thin cloth between IMU and skin (sweat protection) / IMU和皮肤之间隔一层薄布防汗
- Take a photo of placement for each session / 每次拍照留档

---

## 四、Current Data Composition / 目前数据组成

### 4.1 Data Summary / 数据总览

| Category 类别 | Subject 被试 | Count 数量 | Duration 时长 | Hz |
|--------------|-------------|-----------|--------------|-----|
| Trip (绊倒) | Hang | 20 | ~70s each | 157-163 |
| Trip (绊倒) | Xiaoyang | 8 | ~70s each | 161-163 |
| Normal Walking (正常走路) | Hang | 10 | ~70s each | 163-164 |
| Normal Walking (正常走路) | Xiaoyang | 10 | ~70s each | 161-163 |
| **Total 合计** | | **49** | **~50 min** | |

### 4.2 Data Format / 数据格式

Each trial is one CSV file with 32 columns:
每个trial是一个CSV文件，32列：

| Column 列 | Description 说明 | Unit 单位 |
|-----------|-----------------|-----------|
| timestamp | Time (millis or NTP epoch) / 时间戳 | seconds |
| thigh_acc_x/y/z | Thigh accelerometer / 大腿加速度 | m/s² |
| thigh_gyro_x/y/z | Thigh gyroscope / 大腿角速度 | rad/s |
| thigh_mag_x/y/z | Thigh magnetometer / 大腿磁力计 | µT |
| shank_acc_x/y/z | Shank accelerometer / 小腿加速度 | m/s² |
| shank_gyro_x/y/z | Shank gyroscope / 小腿角速度 | rad/s |
| shank_mag_x/y/z | Shank magnetometer / 小腿磁力计 | µT |
| foot_acc_x/y/z | Foot accelerometer / 脚加速度 | m/s² |
| foot_gyro_x/y/z | Foot gyroscope (converted) / 脚角速度（已转换） | rad/s |
| foot_mag_x/y/z | Foot magnetometer / 脚磁力计 | µT |
| foot_euler_x/y/z | Foot Euler angles (BNO055 fusion) / 脚欧拉角 | degrees |
| perturbation_event | 0=normal, 1=perturbation marked / 标记 | binary |

### 4.3 Naming Convention / 命名规则

```
[scenario]_[subject]_[number].csv

scenario: slip / trip / normal
subject:  hang / xiaoyang
number:   01, 02, 03, ...

Examples:
  slip_hang_01.csv      — Hang's 1st trip trial
  normal_xiaoyang_03.csv — Xiaoyang's 3rd normal walking trial
```

### 4.4 Key Notes / 重要说明

- **BNO055 gyro**: converted to rad/s in firmware (÷57.2958) / 已在固件中转换
- **No slip data**: only trip perturbations collected / 只采集了绊倒数据，没有滑倒
- **Thigh/Shank swap fixed**: early data had CH_THIGH=5/CH_SHANK=4 reversed, fixed with `swap_thigh_shank.py` / 早期数据大腿小腿标签反了，已用脚本修复
- **Zero empty fields**: V6 binary firmware sends complete frames / V6二进制固件发送完整帧，无空值
- **Perturbation events mostly unlabeled**: use energy spike detection or video for post-hoc labeling / 大部分未按空格标记，需事后标注

---

## 五、Experiment Protocol / 实验流程

### 5.1 Pre-experiment Setup / 实验前准备

1. Connect laptop to WiFi (same network as ESP32) / 笔记本连WiFi
2. Run `ifconfig en0 | grep inet` to confirm IP / 确认IP
3. Power on ESP32 (USB power bank) / ESP32通电
4. Wait for Serial Monitor to show `# READY` / 等待READY
5. Run `python udp_logger.py --test` to verify 150+ Hz, zero empty fields / 测试确认

### 5.2 Single Trial Flow / 单次Trial流程

```
Timeline / 时间轴:
0s          10-15s        20-40s         50-60s      70s
|            |              |              |          |
Start walk   Normal gait   PERTURBATION   Recovery   Stop
开始走路     正常步态      绊倒事件       恢复       结束
```

Each trial: ~60-80 seconds, one perturbation event per trial.
每个trial约60-80秒，每次只有一个perturbation事件。

### 5.3 Making Trips Natural / 如何让绊倒自然

- Experimenter places obstacle randomly — subject doesn't know exact position / 实验员随机放置障碍物
- Give subject a distraction task (e.g. count backwards from 100 by 7) / 让被试做分心任务
- Mix in normal trials — subject doesn't know if obstacle is present / 穿插normal trial
- Use different obstacle heights (1cm, 2cm, 3cm) / 不同高度障碍物
- Walking and stopping after trip is natural — don't force constant speed / 绊倒后走走停停是正常的

### 5.4 Data Collection Plan / 采集计划

| Scenario 场景 | Hang | Xiaoyang | Total 合计 |
|--------------|------|----------|-----------|
| Trip (绊倒) | 20 | 20 | 40 |
| Normal Walking (正常走路) | 10 | 10 | 20 |
| **Total 合计** | **30** | **30** | **60** |

Time estimate: ~2.5 hours total / 预计2.5小时

---

## 六、Visualization Guide / 可视化指南

### 6.1 How to Run / 使用方法

```bash
# All plots for one trial (saved to figures/[trial_name]/)
# 单个trial全套图（保存到 figures/[trial名]/ 文件夹）
python visualize_all.py data/slip_hang_11.csv --save

# Quick mode — only 3 key plots (dashboard, magnitudes, jerk)
# 快速模式——只出3张关键图
python visualize_all.py data/slip_hang_11.csv --save --quick

# Batch all data
# 批量处理所有数据
python visualize_all.py data/ --batch --save

# Compare trip vs normal (side-by-side)
# 绊倒 vs 正常对比
python visualize_all.py --compare data/slip_hang_11.csv data/normal_hang_01.csv --save

# Overlay multiple trips
# 叠加多条绊倒数据
python visualize_all.py data/slip_hang_09.csv data/slip_hang_10.csv --overlay --save
```

### 6.2 Plot Descriptions / 各图说明

Each trial generates up to 11 plots, saved in `figures/[trial_name]/`:
每个trial最多生成11张图，保存在 `figures/[trial名]/` 文件夹：

#### 📊 Dashboard (`dashboard.png`)
**One-page summary of all key metrics / 一页总览所有关键指标**

4 rows: Row 1 = Acc XYZ per sensor; Row 2 = Gyro XYZ per sensor; Row 3 = Acc & Gyro magnitudes; Row 4 = Jerk + Euler angles. Best plot for quick quality check.

四行：第一行各传感器加速度XYZ；第二行角速度XYZ；第三行加速度和角速度幅值；第四行Jerk和欧拉角。最适合快速检查数据质量。

#### 📈 Overview (`overview.png`)
**6-panel raw XYZ data / 6面板原始XYZ数据**

Shows acc XYZ and gyro XYZ for each sensor separately. Use this to check individual axis behavior and identify which axis responds most during perturbation.

分别显示每个传感器的加速度和角速度XYZ分量。可以看哪个轴在perturbation时变化最大。

#### 📈 Magnitudes (`magnitudes.png`)
**Acceleration and gyroscope magnitude for all 3 sensors / 三个传感器的加速度和角速度幅值**

Top: acc magnitude (should hover around 9.81 during rest, spike during perturbation). Bottom: gyro magnitude. Most intuitive plot — perturbation is a clear spike above the walking baseline. FOOT usually shows largest response because it contacts the obstacle directly.

上：加速度幅值（静止时≈9.81，perturbation时飙升）。下：角速度幅值。最直观的图——perturbation表现为明显的spike。FOOT响应通常最大因为直接接触障碍物。

#### 📈 Jerk (`jerk.png`)
**Rate of change of acceleration / 加速度变化率**

Jerk = d(acceleration)/dt. Measures how "sudden" a motion is. Perturbation causes extremely high jerk values. Very noisy during walking (normal), but the perturbation spike is denser and taller. Better used as model feature than for visual inspection.

Jerk = 加速度的时间导数。衡量运动的"突然程度"。走路时本来就很嘈杂，但perturbation时spike更密集更高。更适合作为模型特征而非人眼判断。

#### 📈 Energy (`energy.png`)
**500ms sliding window signal energy / 500毫秒滑动窗口信号能量**

Acc energy = mean of (acc_mag - gravity)² over 500ms window. Best plot for identifying perturbation timing — shows up as a sharp, isolated peak. During normal walking, energy is low and periodic. During perturbation, energy spikes dramatically (10-50x above walking baseline).

加速度能量 = 500ms窗口内(加速度幅值-重力)²的均值。最适合定位perturbation时间——表现为尖锐孤立的peak。正常走路时能量低且有周期性，perturbation时能量飙升10-50倍。

#### 🎵 Spectrogram (`spectrogram.png`)
**Time-frequency view of FOOT acc_z / 脚部加速度的时频图**

Top: time-domain signal. Bottom: spectrogram showing how frequency content changes over time. Normal walking shows rhythmic bands at 0-5Hz (gait frequency). Perturbation appears as a bright broadband burst (0-30Hz) — energy across all frequencies simultaneously.

上：时域波形。下：频谱随时间变化的热力图。正常步态在0-5Hz有规律条纹。Perturbation表现为亮黄色宽带burst（0-30Hz全频段同时有能量）。

#### 🔗 Correlation (`correlation.png`)
**Rolling correlation between sensor pairs / 传感器对之间的滚动相关性**

Measures how synchronized the 3 sensors' movements are over a 0.7s window. During normal walking, sensors are moderately correlated (0.3-0.8). During perturbation, correlation can drop suddenly as body segments decouple (e.g., foot stops but thigh continues forward).

衡量三个传感器运动的同步性。正常走路时中度相关（0.3-0.8）。Perturbation时相关性突然下降——身体各部位运动不再协调（如脚被挡住但大腿继续前进）。

#### 🦶 Joint Angles (`joints.png`)
**Foot Euler angles from BNO055 onboard fusion / BNO055板载融合输出的脚部欧拉角**

Shows foot roll and pitch over time. Large jumps indicate perturbation or foot orientation changes. Note: only available for FOOT sensor (BNO055 has onboard fusion; ICM-20948 does not).

显示脚的roll和pitch随时间变化。大幅跳变表示perturbation或脚部姿态变化。注意只有FOOT传感器有此数据。

#### 📊 FFT (`fft_thigh.png`)
**Frequency spectrum of thigh acc_z / 大腿加速度Z轴频谱**

Shows where signal energy is concentrated. For walking + perturbation data, energy is typically concentrated at 0-10Hz. Above 15Hz is mostly noise. This confirms our sampling rate (150+Hz) is more than sufficient.

显示信号能量的频率分布。步态+perturbation数据的能量集中在0-10Hz，15Hz以上基本是噪声。证实我们的采样率（150+Hz）完全够用。

#### 📉 Downsample (`downsample.png`)
**Same data at different sampling rates / 同一数据在不同采样率下的对比**

Compares thigh acc magnitude at original Hz, 100Hz, 80Hz, and 50Hz. Used to evaluate if lower sampling rates lose important perturbation details. In practice, even 50Hz preserves the perturbation spike — the peak value and timing remain consistent.

对比大腿加速度幅值在原始频率、100Hz、80Hz、50Hz下的表现。用于评估降低采样率是否丢失重要信息。实际上50Hz也能保留perturbation的spike。

#### 🌐 3D Trajectory (`3d.png`)
**Acceleration vector in 3D space, colored by time / 加速度在3D空间的轨迹，按时间着色**

Normal walking forms a tight cluster. Perturbation causes outlier points that fly far from the main cluster. Color indicates time — you can see when the outlier happened. Good for presentations but less informative than other plots.

正常走路形成紧密的点团。Perturbation时有点飞出很远。颜色表示时间。适合放在presentation里，但信息量不如其他图。

### 6.3 Comparison Plots / 对比图

#### Trip vs Normal (`trip_vs_normal.png`)
**Side-by-side comparison of trip trial and normal walking / 绊倒与正常走路并排对比**

3 rows: acc magnitude, gyro magnitude, jerk. Left = trip, right = normal. Key differences:
- Trip has a distinct spike in acc and gyro that normal lacks
- Trip's gyro magnitude shows FOOT >> THIGH/SHANK at perturbation
- Normal walking has consistent, periodic patterns throughout

三行：加速度幅值、角速度幅值、jerk。左=绊倒，右=正常。关键差异：绊倒有明显的acc和gyro spike，FOOT的gyro远大于THIGH/SHANK；正常走路全程有规律的周期性模式。

#### Trial Overlay (`overlay.png`)
**Multiple trials superimposed for comparison / 多条trial叠加对比**

Shows thigh acc and gyro magnitude from different trials on the same axes. Useful for checking consistency across trials — perturbation spikes should be visible in each trip trial at different times, while walking baseline should be similar across all trials.

在同一坐标轴上叠加多条trial的大腿加速度和角速度幅值。用于检查不同trial之间的一致性——每条trip trial应该在不同时间有perturbation spike，走路的baseline应该相似。

---

## 七、Data Cleaning / 数据清理

```bash
# Clean all data — fixes gyro units, removes outliers
# 清理所有数据——修复gyro单位，去除异常值
python clean_data.py data/

# Clean with low-pass filter
# 清理 + 低通滤波
python clean_data.py data/ --lowpass
```

What it does / 功能:
1. **Gyro unit check**: auto-detects if BNO055 gyro is in °/s and converts to rad/s / 自动检测并转换gyro单位
2. **Anomaly detection**: flags acc > 50 m/s² or gyro > 20 rad/s / 标记异常值
3. **Fix outliers**: replaces with linear interpolation / 用插值替换异常值
4. **Magnetometer & Euler**: NOT checked (mag spikes from environment are normal, Euler 0-360° is valid range) / 磁力计和欧拉角不检测

Output saved to `cleaned/` folder / 输出到 `cleaned/` 文件夹

---

## 八、Safety / 安全注意事项

1. Place yoga mats near obstacle area / perturbation区域放瑜伽垫
2. Experimenter stands nearby, ready to catch / 实验员站在旁边随时扶住
3. Only mild stumbles needed — no actual falls / 只需轻微绊倒，不需要真摔
4. Wear proper shoes (no slippers/heels) / 穿合适的鞋
5. Rest every 10-15 trials / 每10-15个trial休息
6. Clear walkway of unintended obstacles / 清理场地多余障碍物

---

## 九、File Structure / 文件结构

```
BE124-fall-detection-pipeline/
├── data/                          # Raw CSV files / 原始数据
│   ├── slip_hang_01.csv
│   ├── normal_xiaoyang_01.csv
│   └── ...
├── cleaned/                       # Cleaned data (auto-generated) / 清理后数据
├── figures/                       # Visualization outputs / 可视化图表
│   ├── slip_hang_11/              # Per-trial folder / 每个trial一个文件夹
│   │   ├── dashboard.png
│   │   ├── magnitudes.png
│   │   ├── jerk.png
│   │   ├── overview.png
│   │   ├── energy.png
│   │   ├── spectrogram.png
│   │   ├── correlation.png
│   │   ├── joints.png
│   │   ├── fft_thigh.png
│   │   ├── downsample.png
│   │   └── 3d.png
│   ├── trip_vs_normal.png
│   └── overlay.png
├── videos/                        # Experiment recordings / 实验录像
├── models/                        # Trained models (later) / 模型（后续）
├── firmware/                      # V6.1 ESP32 firmware / 固件
├── udp_logger.py                  # Data collection script / 采集脚本
├── visualize_all.py               # Merged visualization script / 合并可视化脚本
├── clean_data.py                  # Data cleaning script / 数据清理脚本
├── swap_thigh_shank.py            # Fix thigh/shank label swap / 修复标签
└── README.md                      # This file / 本文件
```

---

## 十、Post-Collection Pipeline / 采集后流程

1. **Quality check**: `python visualize_all.py data/ --batch --save --quick` / 质量检查
2. **Clean data**: `python clean_data.py data/` / 清理数据
3. **Label perturbations**: use energy plots or video to mark exact perturbation times / 标注perturbation时间
4. **Preprocessing**: resample to uniform Hz, sliding window (500ms, 250ms stride) / 统一采样率，滑动窗口切片
5. **Model training**: GRU baseline → 1D CNN → CNN+GRU / 模型训练
6. **Analysis**: accuracy vs. prediction horizon, SHAP feature importance, joint angle visualization / 分析
7. **Report + Presentation** / 写报告和presentation

---

## 十一、FAQ / 常见问题

**Q: ESP32 won't connect to WiFi?**
A: Check SSID/password. Restart ESP32. Ensure laptop is on same network. iPhone hotspot: enable "Maximize Compatibility" (forces 2.4GHz).

**Q: udp_logger.py receives no data?**
A: Run `ifconfig en0 | grep inet`, confirm IP matches firmware's `LAPTOP_IP`. Check firewall settings.

**Q: Sampling rate drops over time / UDP drops increasing?**
A: WiFi congestion. Add `delayMicroseconds(500)` to firmware loop. Or reduce devices on network.

**Q: BNO055 data looks wrong?**
A: Recalibrate — move sensor in figure-8 until Serial Monitor shows stable values.

**Q: Forgot to press space for perturbation marker?**
A: No problem. Use energy plot to find the spike, or check video timestamp. The acc/gyro spike is unmistakable.

**Q: Walking pattern changes after trip (stopping, slowing down)?**
A: This is natural biomechanical response. Don't force constant speed. The model should learn real recovery patterns.