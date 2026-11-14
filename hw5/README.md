# 第五次作业：相机标定 + PnP 位姿解算 + 卡尔曼滤波

在第四次作业（装甲板识别）的基础上，打通「二维图像 → 三维定位 → 运动估计」的完整链路。

```
相机采集 → 装甲板识别 → 角点提取 → solvePnP → 卡尔曼滤波 → 运动预测
  ✅已有      ✅已有      ✅已有     新增        新增        新增
```

## 交付物

| 交付物 | 位置 | 说明 |
| --- | --- | --- |
| 源码 | `calibrate.cpp` / `main.cpp` | 标定工具 + 识别/PnP/卡尔曼主程序 |
| 实验记录 | [`docs/experiment-record.docx`](docs/experiment-record.docx) | 标定结果、角点顺序判定、PnP、卡尔曼数据与问题排查 |
| 演示视频 | [`docs/demo.webm`](docs/demo.webm) | 真实相机实时运行：检测框、坐标轴、X/Y/Z 距离、raw/filt/pred |
| 曲线脚本 | `plot_pose.py` | 把 `pose_log.csv` 画成 raw/filt/pred 三联图 |

## 文件

| 文件 | 说明 |
| --- | --- |
| `calibrate.cpp` | 相机标定工具，采集棋盘格图像并输出 `camera.yaml` |
| `main.cpp` | 主程序：识别 + PnP + 卡尔曼 + 可视化 |
| `CMakeLists.txt` | 同时构建 `Calibrate` 和 `ArmorPnP` 两个可执行文件 |
| `plot_pose.py` | 读取 `pose_log.csv`，画出滤波前后对比图 |

## 编译

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

产物：`build/Calibrate`、`build/ArmorPnP`

## 使用流程

### 第一步：相机标定

**先改 `calibrate.cpp` 顶部的棋盘格参数**（必须和你手上那块板子一致）：

```cpp
static int   BOARD_COLS = 9;      // 内角点列数（不是方格数！）
static int   BOARD_ROWS = 6;      // 内角点行数
static float SQUARE_MM  = 26.0f;  // 单个方格边长 mm（实测）
```

> 一块 10×7 方格的板子 = 9×6 个**内角点**。

```bash
sudo ./Calibrate
# 或指定参数
sudo ./Calibrate --cols 9 --rows 6 --square 25
```

操作：
1. 举着棋盘格在相机前变换姿态，**覆盖：左/中/右、上/中/下、远/近、正对/倾斜**
2. 画面出现彩色角点 + 绿色框 = 识别成功，按 **空格** 保存一帧
3. 采集 **15~20 张** 后按 **ESC** 结束
4. 程序自动标定，输出内参、畸变、重投影误差，写入 `camera.yaml`

合格标准：**RMS 重投影误差 < 0.5 px**。如果大于 1 px，说明采样的角度覆盖不够，重新采。

### 第二步：运行位姿解算

**先改 `main.cpp` 顶部的装甲板尺寸**（用尺子量实物）：

```cpp
static const float ARMOR_W_MM = 224.0f;   // 宽（实测）
static const float ARMOR_H_MM = 126.0f;   // 高（实测）
```

```bash
sudo ./ArmorPnP --exposure 10000 --gain 0
```

画面显示：
- 检测框 + 关键点（**每个角点标了序号 0~3**，用来核对角点顺序）
- 装甲板坐标系（X 红 / Y 绿 / Z 蓝）
- 左下角：`X Y Z` 坐标（mm）、距离（mm / m）
- `raw` 原始测量 / `filt` 滤波后 / `pred` 预测位置

终端每 30 帧输出一行：

```
[pose] raw(  12.3,  -45.6,  812.4) filt(  11.8,  -45.1,  810.9) pred(  11.5,  -44.9,  810.2) dist 813.7 mm
```

同时写入 `pose_log.csv`，用于画滤波前后对比曲线：

```
t_ms,detected,raw_x,raw_y,raw_z,filt_x,filt_y,filt_z,pred_x,pred_y,pred_z
```

## 参数说明

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `ARMOR_W_MM` / `ARMOR_H_MM` | main.cpp | 装甲板真实尺寸，**直接影响 Z 与距离** |
| `USE_KEYPOINTS` | main.cpp | true 用模型关键点；false 用检测框四角（退路） |
| `KF_MEAS_NOISE` | main.cpp | 测量噪声，越大越信模型（更平滑但更迟钝） |
| `KF_PROC_NOISE` | main.cpp | 过程噪声，越大越信测量（更灵敏但更抖） |
| `PREDICT_HORIZON_S` | main.cpp | 预测提前量，默认 0.15 s |

### 调卡尔曼的经验

- **滤波后抖得厉害** → 增大 `KF_MEAS_NOISE`（比如 4 → 20）
- **滤波后跟不上目标**（有明显的滞后感）→ 减小 `KF_MEAS_NOISE` 或增大 `KF_PROC_NOISE`
- 默认 `4.0 / 0.8` 是 mm 单位下比较中庸的值

## 角点顺序

PnP 要求二维角点和三维物点**一一对应**，顺序错了位姿会翻掉（Z 变负、坐标乱飞）。

本程序假定的顺序：

```
0: 左上 (-w/2, -h/2)    1: 右上 ( w/2, -h/2)
3: 左下 (-w/2,  h/2)    2: 右下 ( w/2,  h/2)
```

画面上每个角点旁标了序号。**如果 0 号不在左上角**，有两种改法：
1. 改 `makeArmorObjectPoints()` 里的三维点顺序，去迁就模型；
2. 或者在取到 `corners` 之后做一次循环移位。

判断标准：**Z 恒为正、数值平稳、坐标轴四条线跟着装甲板转**。

## 常见问题

**Q：`[CAL] 没找到 camera.yaml`**
先跑 `./Calibrate` 完成标定，且 `camera.yaml` 要和可执行文件在同一目录。

**Q：距离明显不对**
装甲板尺寸填错。Z ≈ f × 实际宽度 / 图像中的像素宽度，尺寸差一倍，Z 就差一倍。

**Q：Z 是负的 / 位姿乱跳**
角点顺序不对，见上面「角点顺序」。

**Q：标定 RMS 很大（> 1px）**
采样角度太集中。要覆盖画面四角、不同倾斜角，且棋盘格在每个位置都完整可见。
