# BE124 人工反射弧项目 — 实验采集手册

## 一、实验目标

用3个IMU（大腿、小腿、脚）采集步态数据，训练模型预测perturbation（摔倒前兆）。

## 二、硬件清单

| 物品 | 数量 | 说明 |
|------|------|------|
| ESP32-S3 Feather | 1 | 主控板 |
| PCA9548A Multiplexer | 1 | I2C多路复用 |
| ICM-20948 IMU | 2 | 大腿 + 小腿 |
| BNO055 IMU | 1 | 脚 |
| 绑带/魔术贴 | 3组 | 固定IMU到腿上 |
| USB充电宝 | 1 | 给ESP32供电 |
| 小腰包/布袋 | 1 | 装ESP32+multiplexer+充电宝 |
| 手机 | 1 | 开热点 + 录视频 |
| 笔记本 | 1 | 连热点，跑udp_logger.py |
| 木板/block | 1 | slip场景用 |
| 地毯条/胶带凸起 | 1 | trip场景用 |

## 三、IMU安装位置

全部安装在 **dominant leg（惯用腿）**：

```
         ┌──────┐
         │ 大腿 │ ← ICM-20948 #1，绑在大腿外侧中段
         └──┬───┘
            │
         ┌──┴───┐
         │ 小腿 │ ← ICM-20948 #2，绑在小腿胫骨外侧中段
         └──┬───┘
            │
         ┌──┴───┐
         │  脚  │ ← BNO055，绑在鞋面/脚背
         └──────┘
```

安装要点：
- IMU的XYZ轴方向三个传感器尽量保持一致
- 绑紧，不能晃动，否则数据有噪声
- 每次实验前检查绑带松紧是否一致
- 拍一张安装位置的照片留档

## 四、实验前准备

### 4.1 固件检查

确认固件是V6版本（Binary UDP），WiFi和IP已配置正确：

```cpp
const char* WIFI_SSID     = "你的热点名";
const char* WIFI_PASSWORD = "你的热点密码";
const char* LAPTOP_IP     = "笔记本IP";
```

确认BNO055 gyro单位已转换（固件里应有 `/57.2958f`）。

### 4.2 开机顺序

1. 手机开热点
2. 笔记本连热点
3. 终端跑 `ifconfig en0 | grep inet` 确认IP
4. 给ESP32通电（充电宝）
5. 等Serial Monitor显示 `# READY`
6. 笔记本跑 `python udp_logger.py --test` 确认数据正常（144Hz，3个传感器都有数据）

### 4.3 BNO055校准

每次开机后，拿着传感器在空中慢慢画8字形。Serial Monitor里会每100帧打印一次，等加速度数据稳定后再开始实验。

## 五、实验场景

### 场景A：Block Slip（滑倒）

```
起点                    block位置                终点
  |________________________|_______________________|
  
  被试从起点走 → 踩到block脚滑 → 恢复后走到终点
```

- 在地板上放一块木板或光滑material
- 被试正常走路，dominant leg踩上去会滑一下
- block位置可以每次稍微变化，让被试不完全预期

### 场景B：Carpet Trip（绊倒）

```
起点                  凸起位置                  终点
  |________________________|_______________________|
  
  被试从起点走 → 脚碰到凸起绊一下 → 恢复后走到终点
```

- 用胶带粘一段地毯边缘或做一个小凸起
- 被试走路时dominant leg被绊

### 场景C：Normal Walking（正常行走）

```
起点                                           终点
  |______________________________________________|
  
  被试正常走路，无任何障碍物
```

- 和perturbation trial走一样的路线
- 步速尽量和perturbation trial一致

## 六、单次Trial流程

### 每个trial 60秒，流程如下：

```
时间轴：
0s          10-15s        20-40s         50s        60s
|            |              |             |          |
开始走路    正常步态     perturbation     恢复走路   结束
            建立baseline  （slip/trip）
```

### 操作步骤：

1. **被试**：站在起点，IMU绑好，准备好
2. **实验员**：
   - 打开手机录像，对准walkway
   - 终端运行（用语音命令提示被试）：
     ```bash
     python udp_logger.py --trial [场景]_[被试]_[编号]
     ```
   - 命名规则举例：
     - `slip_hang_01` — 第1次slip，被试是Hang
     - `trip_xiaoyang_05` — 第5次trip，被试是Xiaoyang
     - `normal_hang_03` — 第3次normal walking，被试是Hang
3. **被试**：听到"开始"后自然走路
4. **perturbation发生时**：实验员观察到后按空格（不用很准，后面可以从数据/视频里精确定位）
5. **60秒后**：按Q停止
6. **休息30秒**，准备下一个trial

## 七、数据采集计划

### 7.1 采集量

| 场景 | Hang | Xiaoyang | 合计 |
|------|------|----------|------|
| Block Slip | 15  | 15 | 30 |
| Carpet Trip | 15 | 15 | 30 |
| Normal Walking | 10 | 10 | 20 |
| **合计** | **40** | **40** | **80** |

### 7.2 时间估算

- 每个trial：60秒采集 + 30秒准备 = 1.5分钟
- 每人40个trial：60分钟
- 两个人加上切换：约2.5小时
- 建议分两次采集，一次约1.5小时

### 7.3 采集顺序建议

不要把同一种场景连续做完，交替做可以减少疲劳和适应效应：

```
第一轮（Hang做被试，Xiaoyang做实验员）：
  slip_hang_01 → normal_hang_01 → trip_hang_01 →
  slip_hang_02 → normal_hang_02 → trip_hang_02 →
  ... 重复到各场景完成

休息10分钟

第二轮（Xiaoyang做被试，Hang做实验员）：
  slip_xiaoyang_01 → normal_xiaoyang_01 → trip_xiaoyang_01 →
  ...
```

## 八、安全注意事项

1. **防护垫**：在perturbation区域旁边放瑜伽垫或软垫，防止真的摔伤
2. **旁边有人**：实验员站在被试旁边，随时准备扶住
3. **控制强度**：perturbation只需要轻微的滑/绊，不需要真的摔倒。一个小的stumble就够了
4. **穿合适的鞋**：不要穿拖鞋或高跟鞋
5. **疲劳休息**：每10-15个trial休息5分钟
6. **场地清理**：确保walkway除了设计的obstacle外没有其他绊倒风险

## 九、数据文件管理

### 文件夹结构

```
BE124_Data_Collection/
├── data/                          # 所有原始CSV
│   ├── slip_hang_01.csv
│   ├── slip_hang_02.csv
│   ├── trip_hang_01.csv
│   ├── normal_hang_01.csv
│   └── ...
├── videos/                        # 对应的录像
│   ├── slip_hang_01.mp4
│   └── ...
├── processed/                     # 预处理后的数据（自动生成）
├── figures/                       # 可视化图表（自动生成）
├── models/                        # 模型训练结果（后续）
├── firmware/                      # V6固件
├── udp_logger.py                  # 数据采集脚本
├── visualize.py                   # 可视化脚本
└── README.md                      # 本文件
```

### 命名规则

```
[场景]_[被试]_[编号].csv

场景：slip / trip / normal
被试：hang / xiaoyang
编号：01, 02, 03, ...
```

## 十、采集后质量检查

每完成一批trial（比如每10个），跑一下可视化确认数据质量：

```bash
# 快速检查最新几个trial
python visualize.py data/slip_hang_01.csv --save

# 批量检查所有数据
python visualize.py data/ --batch --save
```

检查要点：
- Hz是否稳定在~144
- 三个传感器都有数据（无空值）
- 加速度baseline在9.8附近
- perturbation时刻有明显的信号变化
- gyro单位是否统一（FOOT和THIGH/SHANK应该在同一量级）

## 十一、Perturbation标注

### 粗标注（采集时）

按空格大概标记。不用很准。

### 精标注（采集后）

1. 打开CSV和对应视频
2. 视频里找到perturbation发生的精确时刻
3. 对应CSV里的NTP时间戳（epoch秒），在perturbation_event列标记为1
4. 标记perturbation前后各500ms的范围作为positive window

### 自动标注（推荐）

后续写一个脚本：自动在加速度/角速度信号里找异常spike，定位perturbation时刻。这比人工标注更准更快。

## 十二、采集后下一步

1. **数据预处理**：统一单位、降采样实验、sliding window切片
2. **模型训练**：GRU baseline → 1D CNN → CNN+GRU
3. **分析**：
   - Accuracy vs. prediction horizon 曲线
   - Feature importance (SHAP)
   - Joint angle 可视化
   - Slip vs. trip 对比
4. **写Report + 做Presentation**

## 十三、常见问题

**Q: ESP32连不上WiFi？**
A: 确认热点名密码对不对，手机热点是否开着。重启ESP32试试。

**Q: udp_logger.py收不到数据？**
A: 确认笔记本和ESP32连的是同一个热点。跑ifconfig检查IP是否和固件里的LAPTOP_IP一致。

**Q: 采样率突然下降？**
A: 可能WiFi信号不好。把笔记本和手机放近一点。或者检查充电宝电量。

**Q: BNO055数据异常？**
A: 可能需要重新校准。重启ESP32，画8字形校准。

**Q: 忘记按空格标记了？**
A: 没关系，事后看视频/数据标注。加速度spike很明显，不会漏。

**Q: 被试太适应了，perturbation反应不自然？**
A: 随机变换block/凸起的位置，或者加一些normal trial穿插，让被试不确定哪次有obstacle。
