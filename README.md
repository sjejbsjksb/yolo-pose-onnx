# 海康工业相机 + YOLO 装甲板实时检测

计算机视觉课程第四次大作业：使用**海康工业相机实时取流**，通过 **OpenCV DNN** 加载
**YOLOv8-Pose** 模型，在 C++ 中完成装甲板检测、关键点回归与实时可视化。

本仓库同时包含**模型权重**、**C++ 源码**、**模型修补工具**与**实验文档**。

---

## 一、实验环境

| 项目 | 配置 |
| --- | --- |
| 宿主机 | Windows 11 |
| 虚拟机 | VMware Workstation + Ubuntu 22.04 LTS |
| 相机 | 海康 **MV-CS016-10UC**（USB3） |
| 相机序列号 | DA4302122 |
| 相机固件 | V2.1.4 230927 1116552 |
| 相机 SDK | 海康 MVS SDK for Linux（`/opt/MVS`） |
| 视觉库 | **OpenCV 4.6.0**（含 dnn 模块） |
| 推理 | OpenCV DNN，CPU 多线程（虚拟机无 GPU） |
| 构建 | CMake 3.16+ / C++17 / GCC |
| 采集分辨率 | 1440 × 1080（全幅面，未做 ROI 裁剪） |

---

## 二、目录结构

```
yolo-pose-onnx/
├── best.onnx                     原始导出模型（640 输入，需 OpenCV ≥ 4.8）
├── best_opencv46.onnx            640 输入 + OpenCV 4.6 兼容补丁
├── best_480_opencv46.onnx        480 输入 + OpenCV 4.6 兼容补丁（推荐）
├── cpp/
│   ├── main.cpp                  完整 C++ 程序（取流 → 推理 → 显示）
│   └── CMakeLists.txt            构建脚本
├── docs/
│   ├── experiment-record.docx    实验记录（环境、流程、三组对照实验数据）
│   └── error-handling-record.docx 错误处理与资源释放记录
├── demo/
│   └── demo.mp4                  运行演示视频（真实相机实时检测）
├── tools/
│   └── patch_onnx_opencv46.py    把模型修补成 OpenCV 4.6 可用
└── README.md
```

## 三、交付文档与演示

| 交付物 | 位置 | 内容 |
| --- | --- | --- |
| 实验记录 | [`docs/experiment-record.docx`](docs/experiment-record.docx) | 实验环境、系统设计、关键实现、**曝光/增益/图像格式三组对照实验数据与分析**、问题排查、结论 |
| 错误处理记录 | [`docs/error-handling-record.docx`](docs/error-handling-record.docx) | 5 个异常场景的实测日志、资源释放检查表、Ctrl+C 句柄泄漏问题的发现与修复 |
| 运行演示视频 | [`demo/demo.mp4`](demo/demo.mp4) | 真实工业相机实时检测装甲板，含检测框、置信度、关键点、FPS 与中心点坐标 |

---

## 四、模型权重说明

### 4.1 格式与来源

| 项目 | 说明 |
| --- | --- |
| 格式 | ONNX（opset 12，静态 batch，IR version 7） |
| 来源 | Ultralytics YOLOv8-Pose，由 `best.pt` 导出 |
| 输入 | `images`，`[1, 3, 640, 640]` 或 `[1, 3, 480, 480]`，RGB，float32，0~1 |
| 输出 | `output0`，`[1, 18, 8400]`（640）或 `[1, 18, 4725]`（480） |

- 8400 = 80² + 40² + 20²（640 输入下三个特征层锚点）
- 4725 = 60² + 30² + 15²（480 输入下三个特征层锚点）

### 4.2 输出 18 个通道的含义

`18 = 4 + 6 + 8`，即 **6 个类别 + 4 个关键点**：

| 通道 | 含义 |
| --- | --- |
| 0 ~ 3 | `cx, cy, w, h`，输入坐标系下的像素值 |
| 4 ~ 9 | 6 个类别分数，已过 sigmoid，取值 0~1 |
| 10 ~ 17 | 4 个关键点，每个点只有 `x, y` 两个值，**没有置信度** |

### 4.3 三个版本的选用

| 文件 | 输入 | 适用 OpenCV | 说明 |
| --- | --- | --- | --- |
| `best.onnx` | 640 | ≥ 4.8 | 原始导出 |
| `best_opencv46.onnx` | 640 | ≥ 4.0 | 打了广播补丁 |
| `best_480_opencv46.onnx` | 480 | ≥ 4.0 | 打了补丁 + 降分辨率，帧率约 ×1.8 |

### 4.4 为什么需要补丁

OpenCV 4.6 / 4.7 的 DNN 在计算 Eltwise(Mul/Add) 的广播形状时，
两个输入 rank 差大于 1 会触发内部越界，报错：

```
OpenCV(4.6.0) .../dnn/shape_utils.hpp:171: error: (-215:Assertion failed)
start <= (int)shape.size() && end <= (int)shape.size() && start <= end in function 'total'
```

本模型 pose 解码头里正好有 `[1,4,2,8400] * [1,8400]` 这种广播写法。
官方在 4.8 修复（[opencv/opencv#23464](https://github.com/opencv/opencv/issues/23464)）。

补丁的做法是把这些广播常量**提前展开成完整形状**，从"广播乘"变成"等形状乘"，
数学上完全等价，实测输出与原模型**逐位相同**（max abs diff = 0）。

重新生成：

```bash
pip install onnx numpy
python tools/patch_onnx_opencv46.py best.onnx best_opencv46.onnx
```

### 4.5 重新导出模型

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=480, opset=12, simplify=True)
```

导出后仍需跑一次补丁脚本。

---

## 五、编译

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

CMake 会自动探测 `main.cpp` 的位置（根目录或 `src/`），并查找 `/opt/MVS` 下的
`MvCameraControl.h` 与 `libMvCameraControl.so`。

> 必须用 **Release** 构建。Debug 下 OpenCV 的 `CV_DbgAssert` 全部生效，
> 帧率会明显下降（CMakeLists 中已强制为 Release）。

---

## 六、运行

把模型放到**当前工作目录**（可执行文件所在目录），然后：

```bash
sudo ./YoloCamera                              # 默认参数
sudo ./YoloCamera --exposure 10000 --gain 0    # 指定曝光/增益
sudo ./YoloCamera --pixel bayer                # 切换像素格式
sudo ./YoloCamera --imgsz 480                  # 指定模型输入尺寸
./YoloCamera test.mp4                          # 用视频文件代替相机（调试用）
```

### 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--imgsz N` | 自动探测 | 模型输入尺寸；不填则依次尝试 640/480/416/320 |
| `--exposure N` | 5000 | 曝光时间，单位 µs |
| `--gain N` | 0.0 | 增益，单位 dB |
| `--pixel X` | bgr | 相机输出格式：`bgr` / `rgb` / `bayer` / `mono` |
| 其余参数 | — | 当作视频文件路径，用视频代替相机 |

运行中按 `q` 或 `ESC` 退出，按 `Ctrl+C` 也会走正常释放流程。

---

## 七、程序输出说明

启动时会打印相机身份信息与生效参数：

```
===== hik_yolo_pose =====
[MAIN] 模型输入尺寸: 640 x 640
[MAIN] 曝光 10000 us | 增益 0.0 dB
[YOLO] 模型输入尺寸: 480 x 480（自动探测）
[MVS] 枚举到 1 个设备
[INFO] 相机型号  : MV-CS016-10UC
[INFO] 序列号    : DA4302122
[INFO] 固件版本  : V2.1.4 230927 1116552
[INFO] 像素格式  : bgr
[MAIN] 相机已启动，开始取流
```

运行中每 30 帧输出一次统计：

```
[time] pre 4.0 | infer 111.4 | post 0.1 | ui 15.2 | total 131.2 ms
[stat] fps 7.5 | cpu 320% | convert 2.03 ms | e2e 114.6 ms | 1440x1080
[center] obj1  cls=armor  conf=0.76  center=(682, 652)  box=(6,419,1365x467)
```

| 指标 | 含义 |
| --- | --- |
| `pre` | letterbox + blobFromImage 耗时 |
| `infer` | `net.forward()` 耗时 |
| `post` | 解码 + NMS + 坐标反变换耗时 |
| `ui` | 绘制 + imshow + waitKey 耗时 |
| `convert` | 像素格式转换耗时 |
| `e2e` | 从拿到图像到推理结果输出 |
| `cpu` | 进程 CPU 占用（多核可超 100%） |
| `center` | 检测框中心在原图中的像素坐标 |

---

## 八、实测性能

| 输入尺寸 | pre | infer | post | ui | 合计 | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| 640×640 | 6.2 ms | 213.0 ms | 0.2 ms | 17.8 ms | 237.2 ms | 4.2 |
| 480×480 | 4.0 ms | 111.4 ms | 0.1 ms | 15.2 ms | 131.2 ms | 7.5 |

像素格式转换耗时（1440×1080）：

| 相机输出 | convert | 相对 BGR8 |
| --- | --- | --- |
| BGR8 | 2.03 ms | 基准 |
| Mono8 | 3.82 ms | +88% |
| BayerRG8 | 4.68 ms | +130% |

---

## 九、常见问题

**Q1：`[MVS] 一个相机都没枚举到`**
- 检查 USB 是否插好、是否被其他进程占用（上一次程序没有正常退出）
- 用 `sudo` 运行，或确认 `/etc/udev/rules.d/` 下有 MVS 的规则

**Q2：`Can't read ONNX file: best.onnx`**
- 模型不在当前工作目录。程序按**相对路径**查找，需与可执行文件同目录或从项目根目录运行

**Q3：`forward 失败: shape_utils.hpp ... total`**
- 模型没打补丁，且 OpenCV < 4.8。用 `best_opencv46.onnx` / `best_480_opencv46.onnx`，
  或升级 OpenCV 到 4.8 以上

**Q4：`[YOLO] 640/480/416/320 都跑不通`**
- 同 Q3，模型未打补丁

**Q5：手动设置曝光不生效**
- 相机自动曝光会覆盖手动值，程序已在设置前关闭 `ExposureAuto` / `GainAuto`

**Q6：`QSocketNotifier: Can only be used with threads started with QThread`**
- OpenCV 4.6 链接了 Qt 图形后端产生的警告，不影响功能

---

## 十、许可证

模型由 [Ultralytics](https://github.com/ultralytics/ultralytics) 训练并导出，
其许可证为 AGPL-3.0，使用时请遵守对应条款。
