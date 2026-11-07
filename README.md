# YOLOv8-Pose 装甲板检测模型（ONNX）

计算机视觉课程作业用的权重仓库，模型用于**装甲板检测 + 4 关键点姿态估计**。

## 文件

| 文件 | 大小 | SHA-256 |
| --- | --- | --- |
| [`best.onnx`](best.onnx) | 10,195,887 字节（9.72 MB） | `647DA6080EE3487E408B9EB60846DD3135AEBC91DA26C73C4FF3DACF001BD315` |

## 模型信息

| 项目 | 值 |
| --- | --- |
| 任务 | 目标检测 + 关键点回归（YOLOv8-Pose） |
| 来源 | Ultralytics 导出（PyTorch → ONNX，IR version 7） |
| 输入 | 名称 `images`，形状 `1 × 3 × 640 × 640`，RGB，float32，归一化到 0~1 |
| 输出 | 名称 `output0`，形状 `1 × 18 × 8400` |
| 类别数 | 2 |
| 关键点 | 4 个 |

### 输出维度怎么拆（18 = 4 + 2 + 4×3）

| 通道 | 含义 |
| --- | --- |
| 0 ~ 3 | `cx, cy, w, h`，**640 输入坐标系**下的像素值 |
| 4 ~ 5 | 2 个类别的置信度，取值 0~1 |
| 6 ~ 17 | 4 个关键点，每个占 `x, y, conf` 三个通道 |

> 8400 = 80×80 + 40×40 + 20×20，即 640 输入下三个特征层的锚点总数。

## 快速使用

### Python（OpenCV DNN）

```python
import cv2

net = cv2.dnn.readNetFromONNX("best.onnx")
img = cv2.imread("test.jpg")

blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640),
                             swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()
print(out.shape)          # (1, 18, 8400)
```

> 图片不是正方形时要做 **letterbox**（等比缩放 + 灰边填充），推理后再把框和关键点映射回原图，
> 否则画面会被拉伸，关键点位置也会偏。

### C++（OpenCV DNN）

```cpp
cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");

cv::Mat blob;
cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0, cv::Size(640, 640),
                       cv::Scalar(), true /*swapRB*/, false /*crop*/, CV_32F);

net.setInput(blob);
cv::Mat out = net.forward();   // 1 x 18 x 8400
```

## ⚠️ OpenCV 4.6 / 4.7 兼容性问题

如果你用 **OpenCV 4.6.x 或 4.7.x** 的 DNN 模块跑 `best.onnx`，会遇到：

```
OpenCV(4.6.0) .../dnn/shape_utils.hpp:171: error: (-215:Assertion failed)
start <= (int)shape.size() && end <= (int)shape.size() && start <= end
in function 'total'
```

`readNetFromONNX` 能成功，一 `forward()` 就抛。原因：OpenCV 4.6 的 DNN 在算
Eltwise(Mul/Add) 的广播形状时，两个输入 rank 差超过 1 就会越界。本模型的 pose
解码头里正好有 `[1,4,2,8400] * [1,8400]` 这种写法。
官方在 **4.8** 修好了（[opencv/opencv#23464](https://github.com/opencv/opencv/issues/23464)）。

**两个解法，二选一：**

| 方案 | 做法 | 耗时 |
| --- | --- | --- |
| A. 用修补版模型 | 下载下面的 `best_opencv46.onnx`，重命名成 `best.onnx` 直接用 | 1 分钟 |
| B. 升级 OpenCV | 编译安装 OpenCV ≥ 4.8 | 约 1 小时 |

修补版是把那几个广播常量提前展开成完整形状（数学等价，实测输出**逐位相同**），
体积从 9.72 MB 涨到 10.24 MB。

- `best_opencv46.onnx` — SHA-256：`3684BD0F87038D846964BE155CE8FD020A932EC24EAF0A336E52CF2C23D7AC61`
- 自己重新生成：`python tools/patch_onnx_opencv46.py best.onnx best_opencv46.onnx`（需要 `pip install onnx numpy`）

## 三个模型版本怎么选

| 文件 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `best.onnx` | 640×640 | [1,18,8400] | 原始导出，需要 OpenCV ≥ 4.8 |
| `best_opencv46.onnx` | 640×640 | [1,18,8400] | 打补丁版，OpenCV 4.6 可用 |
| **`best_480_opencv46.onnx`** | **480×480** | **[1,18,4725]** | **打补丁 + 降分辨率，帧率约 ×1.7** |

480 版是用 `best.pt` 重新导出的：

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=480, opset=12, simplify=True)
```

- SHA-256：`33E350F2C743B5AE46ABAC1126CDEA8552A103EEA164DB91943B4128D391A685`
- 输出 4725 = 60² + 30² + 15²（480 输入下三个特征层的锚点数）

C++ 端**不需要手动改 `INPUT_W/H`**——程序启动时会自动探测模型输入尺寸
（先试 `--imgsz` 指定的，再试 640/480/416/320），打印一行确认：

```
[YOLO] 模型输入尺寸: 480 x 480（自动探测）
```

## C++ 客户端（海康工业相机 + OpenCV DNN）

`cpp/` 目录是配套的 C++ 代码，Ubuntu 22.04 + OpenCV + MVS SDK：

| 文件 | 说明 |
| --- | --- |
| `cpp/main.cpp` | 取流 → letterbox → DNN 推理 → 解码/NMS → 画图 → FPS，单文件 |
| `cpp/CMakeLists.txt` | 自动探测 `main.cpp` 位置，目标名 `YoloCamera` |

直接拿下来用：

```bash
cd ~/ClionProjects/YOLO
wget -O main.cpp         https://raw.githubusercontent.com/sjejbsjksb/yolo-pose-onnx/main/cpp/main.cpp
wget -O CMakeLists.txt   https://raw.githubusercontent.com/sjejbsjksb/yolo-pose-onnx/main/cpp/CMakeLists.txt
```

编译运行：

```bash
rm -rf cmake-build-debug && cmake -S . -B build && cmake --build build -j$(nproc)
./build/YoloCamera            # 用相机
./build/YoloCamera test.mp4   # 用视频调试
```

终端每 30 帧会打印分阶段耗时，用来定位帧率瓶颈：

```
[time] pre    3.2 | infer   245.1 | post   2.1 | ui   6.4 | total   256.8 ms
```

## 备注

- 本仓库只存放课程作业的权重文件，不含训练代码与数据集。
- 模型由 [Ultralytics](https://github.com/ultralytics/ultralytics) 导出，其许可证为 AGPL-3.0，使用时请遵守对应条款。
