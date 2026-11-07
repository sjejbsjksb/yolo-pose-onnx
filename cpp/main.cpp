// ============================================================================
//  海康工业相机 + YOLOv8-Pose(ONNX) 实时检测  —— 整合版
//
//  平台 : Ubuntu 22.04 / OpenCV 4.x / MVS SDK for Linux (默认装到 /opt/MVS)
//  模型 : best.onnx         (实测结构，不是猜的)
//           输入 images  : 1 x 3 x 640 x 640   (RGB, /255)
//           输出 output0 : 1 x 18 x 8400
//           18 = 4(box: cx,cy,w,h) + 6(类别分数) + 4*2(关键点 x,y)
//           —— 6 类；4 个关键点，每个只有 x,y，没有 conf
//
//  编译 : ./build.sh
//         或 mkdir build && cd build && cmake .. && make -j$(nproc)
//  运行 : ./build/YoloCamera              # 用相机
//         ./build/YoloCamera test.mp4     # 用视频调试（不接相机也能验模型）
//
//  流程 : 取流 -> 转 BGR -> letterbox -> DNN 推理 -> 解码 + NMS -> 画图 -> 显示
// ============================================================================

#include <MvCameraControl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
//  配置区：要改基本只动这里
// ---------------------------------------------------------------------------
static const char* MODEL_PATH = "best.onnx";

// 模型输入尺寸：必须和导出时的 imgsz 一致！
//   640 的模型保持默认；480 的模型用 ./YoloCamera --imgsz 480
static int         INPUT_W = 640;
static int         INPUT_H = 640;

static const int   NUM_CLS  = 6;   // 分类头 model.23.cv3.*.2.weight 输出通道 = 6
static const int   NUM_KPT  = 4;   // 关键点个数
static const int   KPT_DIMS = 2;   // 每个关键点 2 个值 (x,y)，没有 conf

static const float CONF_THRES = 0.50f;   // 置信度阈值
static const float NMS_THRES  = 0.40f;   // NMS IoU 阈值

// 相机采集参数（可用 --exposure / --gain 覆盖，做对照实验用）
static int   g_exposureUs = 5000;    // 曝光时间 µs
static float g_gain       = 0.0f;    // 增益 dB

static const char* WIN_NAME = "Industrial YOLO";

// ---------------------------------------------------------------------------
//  类别表：名字和颜色都在这儿改（颜色是 BGR，不是 RGB！）
//  名字请照着训练时 data.yaml 里的 names 顺序填，索引就是类别号
// ---------------------------------------------------------------------------
static const char* CLASS_NAMES[NUM_CLS] = {
    "armor",     // 0
    "cls1",      // 1
    "cls2",      // 2
    "cls3",      // 3
    "cls4",      // 4
    "cls5",      // 5
};

static const Scalar CLASS_COLORS[NUM_CLS] = {
    Scalar(0,   0, 255),   // 0 红
    Scalar(255, 0,   0),   // 1 蓝
    Scalar(0, 255,   0),   // 2 绿
    Scalar(0, 255, 255),   // 3 黄
    Scalar(255, 0, 255),   // 4 紫
    Scalar(255, 255, 0),   // 5 青
};

// 框的颜色要不要"跟着目标本身的颜色走"？
//   true  : 从框内像素采样主色（红灯画红框、绿灯画绿框）
//   false : 用上面 CLASS_COLORS 里按类别写死的颜色
// 采样失败（框内太暗/太灰）时自动退回类别色
static const bool USE_OBJECT_COLOR = true;

// ---------------------------------------------------------------------------
//  数据结构
// ---------------------------------------------------------------------------
struct Detection {
    Rect2f          box;
    float           score = 0.f;
    int             cls   = 0;
    vector<Point2f> kpts;
};

struct LetterboxInfo {
    float scale = 1.f;   // 缩放比例
    float padX  = 0.f;   // 左边补了多少像素
    float padY  = 0.f;   // 上边补了多少像素
};

// ---------------------------------------------------------------------------
//  小工具
// ---------------------------------------------------------------------------
static void mvsError(const char* api, int code)
{
    printf("[MVS] %s 失败, code = 0x%X\n", api, (unsigned int)code);
}

// 读相机的字符串属性：型号 / 序列号 / 固件版本
static string camString(void* h, const char* key)
{
    MVCC_STRINGVALUE sv;
    memset(&sv, 0, sizeof(sv));
    sv.nMaxLength = (int64_t)sizeof(sv.chCurValue);
    if (MV_CC_GetStringValue(h, key, &sv) != MV_OK) return string("N/A");
    return string(sv.chCurValue);
}

// 等比缩放 + 灰边填充，保持长宽比（比直接拉伸精度高）
static Mat letterbox(const Mat& src, int newW, int newH, LetterboxInfo& info)
{
    const int w = src.cols;
    const int h = src.rows;
    const float r = std::min((float)newW / (float)w, (float)newH / (float)h);

    const int nw = (int)std::lround(w * r);
    const int nh = (int)std::lround(h * r);

    Mat resized;
    resize(src, resized, Size(nw, nh));

    const int dw = newW - nw;
    const int dh = newH - nh;
    const int top    = dh / 2, bottom = dh - top;
    const int left   = dw / 2, right  = dw - left;

    Mat out;
    copyMakeBorder(resized, out, top, bottom, left, right,
                   BORDER_CONSTANT, Scalar(114, 114, 114));

    info.scale = r;
    info.padX  = (float)left;
    info.padY  = (float)top;
    return out;
}

class Fps
{
public:
    double tick()
    {
        auto now = chrono::steady_clock::now();
        double dt = chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;
        if (dt <= 0.0) return 0.0;
        ema_ = (ema_ <= 0.0) ? dt : (ema_ * 0.85 + dt * 0.15);
        return 1000.0 / ema_;
    }
private:
    chrono::steady_clock::time_point last_ = chrono::steady_clock::now();
    double ema_ = 0.0;
};

// ---------------------------------------------------------------------------
//  海康相机：打开 / 取一帧 BGR / 关闭
// ---------------------------------------------------------------------------
class HikCamera
{
public:
    ~HikCamera() { close(); }

    bool open(int index)
    {
        MV_CC_DEVICE_INFO_LIST list;
        memset(&list, 0, sizeof(list));

        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
        if (ret != MV_OK) {
            mvsError("MV_CC_EnumDevices", ret);
            return false;
        }
        if (list.nDeviceNum == 0) {
            printf("[MVS] 一个相机都没枚举到。检查: 网线/USB、网口 IP 是否同网段、"
                   "是否 sudo 运行、udev 规则是否装好。\n");
            return false;
        }
        printf("[MVS] 枚举到 %u 个设备\n", list.nDeviceNum);
        if (index < 0 || (unsigned int)index >= list.nDeviceNum) {
            printf("[MVS] 相机序号越界: %d\n", index);
            return false;
        }

        ret = MV_CC_CreateHandle(&handle_, list.pDeviceInfo[index]);
        if (ret != MV_OK) {
            mvsError("MV_CC_CreateHandle", ret);
            handle_ = nullptr;
            return false;
        }

        ret = MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0);
        if (ret != MV_OK) {
            mvsError("MV_CC_OpenDevice", ret);
            if ((unsigned int)ret == (unsigned int)MV_E_ACCESS_DENIED)
                printf("[MVS] 权限不足：用 sudo 运行，或检查 /etc/udev/rules.d 下的 MVS 规则。\n");
            close();
            return false;
        }

        // 相机身份信息（作业要求写进实验记录）
        printf("[INFO] 相机型号  : %s\n", camString(handle_, "DeviceModelName").c_str());
        printf("[INFO] 序列号    : %s\n", camString(handle_, "DeviceSerialNumber").c_str());
        printf("[INFO] 固件版本  : %s\n", camString(handle_, "DeviceVersion").c_str());

        // 连续采集
        MV_CC_SetEnumValue(handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);

        // 曝光 / 增益（作业要求；单位 µs，按现场亮度调 1000~20000）
        MV_CC_SetFloatValue(handle_, "ExposureTime", (float)g_exposureUs);
        MV_CC_SetFloatValue(handle_, "Gain", g_gain);

        // GigE 相机设最佳包大小（USB 相机会失败，忽略）
        const int pkt = MV_CC_GetOptimalPacketSize(handle_);
        if (pkt > 0) MV_CC_SetIntValue(handle_, "GevSCPSPacketSize", (unsigned int)pkt);

        ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) {
            mvsError("MV_CC_StartGrabbing", ret);
            close();
            return false;
        }
        grabbing_ = true;
        return true;
    }

    // 阻塞取一帧，转成 BGR。返回的 Mat 指向内部缓冲，下一帧会被覆盖，别存。
    bool grab(Mat& bgr, int timeoutMs = 1000)
    {
        if (!handle_) return false;

        MV_FRAME_OUT fo;
        memset(&fo, 0, sizeof(fo));

        int ret = MV_CC_GetImageBuffer(handle_, &fo, (unsigned int)timeoutMs);
        if (ret != MV_OK) return false;                 // 超时/断流，交给上层处理

        const auto tConv0 = chrono::steady_clock::now();
        lastPixelType = (unsigned int)fo.stFrameInfo.enPixelType;

        const int w = (int)fo.stFrameInfo.nWidth;
        const int h = (int)fo.stFrameInfo.nHeight;
        if (w <= 0 || h <= 0) {
            MV_CC_FreeImageBuffer(handle_, &fo);
            return false;
        }
        lastWidth  = w;
        lastHeight = h;

        const size_t need = (size_t)w * (size_t)h * 3u;
        if (buf_.size() < need) buf_.resize(need);

        // 本来就是 BGR8，直接拷过来
        if (fo.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed) {
            memcpy(buf_.data(), fo.pBufAddr, need);
            MV_CC_FreeImageBuffer(handle_, &fo);
            bgr = Mat(h, w, CV_8UC3, buf_.data());
            lastConvertMs = chrono::duration<double, std::milli>(
                                chrono::steady_clock::now() - tConv0).count();
            return true;
        }

        // 其他格式（Bayer / RGB / Mono）统一转 BGR8
        MV_CC_PIXEL_CONVERT_PARAM cvt;
        memset(&cvt, 0, sizeof(cvt));
        cvt.nWidth         = (unsigned short)w;
        cvt.nHeight        = (unsigned short)h;
        cvt.pSrcData       = fo.pBufAddr;
        cvt.nSrcDataLen    = fo.stFrameInfo.nFrameLen;
        cvt.enSrcPixelType = fo.stFrameInfo.enPixelType;
        cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        cvt.pDstBuffer     = buf_.data();
        cvt.nDstBufferSize = (unsigned int)need;

        ret = MV_CC_ConvertPixelType(handle_, &cvt);
        MV_CC_FreeImageBuffer(handle_, &fo);            // 不管成功失败都要还
        if (ret != MV_OK) {
            mvsError("MV_CC_ConvertPixelType", ret);
            return false;
        }

        bgr = Mat(h, w, CV_8UC3, buf_.data());
        lastConvertMs = chrono::duration<double, std::milli>(
                            chrono::steady_clock::now() - tConv0).count();
        return true;
    }

    // 供实验记录用的采集侧信息
    double lastConvertMs = 0.0;      // 像素格式转换耗时（ms）
    int    lastWidth = 0, lastHeight = 0;
    unsigned int lastPixelType = 0;

    void close()
    {
        if (!handle_) return;
        if (grabbing_) { MV_CC_StopGrabbing(handle_); grabbing_ = false; }
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }

private:
    void*                 handle_   = nullptr;
    bool                  grabbing_ = false;
    vector<unsigned char> buf_;
};

// ---------------------------------------------------------------------------
//  模型：加载 / 推理 / 后处理
// ---------------------------------------------------------------------------
class PoseModel
{
public:
    bool load(const char* path)
    {
        try {
            net_ = dnn::readNetFromONNX(path);
        } catch (const cv::Exception& e) {
            printf("[YOLO] 加载 ONNX 失败: %s\n", e.what());
            return false;
        }
        if (net_.empty()) {
            printf("[YOLO] 网络为空，检查模型路径: %s\n", path);
            return false;
        }
        net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(dnn::DNN_TARGET_CPU);   // 虚拟机没 GPU，老实走 CPU

        // 注意：OpenCV 4.6 没有公开的 Winograd 开关（4.8+ 才有 Net::enableWinograd），
        // 这里只能打开算子融合，属于 4.6 能拿到的全部加速.
        net_.enableFusion(true);

        // ---- 自动探测模型输入尺寸 ----
        // 先试 --imgsz 指定的，再试常见值。用一次空推理探路，抛异常就换下一个。
        vector<int> sizes;
        sizes.push_back(INPUT_W);
        const int common[] = { 640, 480, 416, 320 };
        for (int k = 0; k < 4; ++k) {
            const int s = common[k];
            if (std::find(sizes.begin(), sizes.end(), s) == sizes.end())
                sizes.push_back(s);
        }

        // 探测时会故意喂错尺寸，OpenCV 会往 stderr 刷一大坨报错，先静音
        const utils::logging::LogLevel oldLevel =
            utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

        for (size_t i = 0; i < sizes.size(); ++i) {
            const int s = sizes[i];
            if (i > 0) {                       // 换尺寸 -> 必须重新建网
                net_ = dnn::readNetFromONNX(path);
                net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
                net_.enableFusion(true);
            }
            try {
                int sz[4] = { 1, 3, s, s };
                Mat probe = Mat::zeros(4, sz, CV_32F);
                net_.setInput(probe);
                net_.forward();
                INPUT_W = INPUT_H = s;
                utils::logging::setLogLevel(oldLevel);
                printf("[YOLO] 模型输入尺寸: %d x %d%s\n", s, s,
                       (i == 0 ? "" : "（自动探测）"));
                return true;
            } catch (const cv::Exception&) {
                // 这个尺寸不对，换下一个
            }
        }

        utils::logging::setLogLevel(oldLevel);
        printf("[YOLO] 640/480/416/320 都跑不通，模型有问题或没打补丁（见 README）\n");
        return false;
    }

    // 各阶段耗时（毫秒），每帧刷新，main 里拿去打印
    double msPre   = 0.0;
    double msInfer = 0.0;
    double msPost  = 0.0;

    void infer(const Mat& bgr, vector<Detection>& dets)
    {
        dets.clear();
        if (bgr.empty()) return;

        auto tick0 = chrono::steady_clock::now();

        // ---- 预处理 ----
        LetterboxInfo lb;
        Mat lbImg = letterbox(bgr, INPUT_W, INPUT_H, lb);

        Mat blob;
        dnn::blobFromImage(lbImg, blob, 1.0 / 255.0, Size(INPUT_W, INPUT_H),
                           Scalar(), true /*swapRB*/, false /*crop*/, CV_32F);

        net_.setInput(blob);

        auto tick1 = chrono::steady_clock::now();

        Mat out;
        try {
            out = net_.forward();
        } catch (const cv::Exception& e) {
            printf("[YOLO] forward 失败: %s\n", e.what());
            return;
        }

        auto tick2 = chrono::steady_clock::now();

        msPre   = chrono::duration<double, std::milli>(tick1 - tick0).count();
        msInfer = chrono::duration<double, std::milli>(tick2 - tick1).count();

        // ---- 校验输出 ----
        if (out.dims != 3) {
            printf("[YOLO] 输出维度异常: dims = %d\n", out.dims);
            return;
        }

        const int C = out.size[1];
        const int N = out.size[2];
        const int expectC = 4 + NUM_CLS + KPT_DIMS * NUM_KPT;

        if (C != expectC) {
            printf("[YOLO] 通道数不对: C = %d, 期望 %d (4 + %d + %d*%d)\n",
                   C, expectC, NUM_CLS, KPT_DIMS, NUM_KPT);
            return;
        }

        // [1,18,8400] -> [18,8400] -> [8400,18]
        // 注意 reshape 第二个参数是「行数」，只传一个 int，别写成 {1, C}
        Mat m2d = out.reshape(1, C);   // 18 行 x 8400 列
        Mat rows;
        transpose(m2d, rows);          // 8400 行 x 18 列

        // ---- 逐行解码 ----
        vector<Detection> cand;
        cand.reserve(64);

        for (int i = 0; i < N; ++i) {
            const float* r = rows.ptr<float>(i);

            // 6 类取最大
            float score = -1.0f;
            int   cls   = -1;
            for (int c = 0; c < NUM_CLS; ++c) {
                if (r[4 + c] > score) { score = r[4 + c]; cls = c; }
            }
            if (score < CONF_THRES) continue;

            const float cx = r[0], cy = r[1], bw = r[2], bh = r[3];
            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(bw) || !std::isfinite(bh)) continue;
            if (bw < 1.f || bh < 1.f) continue;

            Detection d;
            d.score = score;
            d.cls   = cls;
            d.box   = Rect2f(cx - 0.5f * bw, cy - 0.5f * bh, bw, bh);

            d.kpts.resize(NUM_KPT);
            for (int k = 0; k < NUM_KPT; ++k) {
                d.kpts[k].x = r[4 + NUM_CLS + KPT_DIMS * k + 0];
                d.kpts[k].y = r[4 + NUM_CLS + KPT_DIMS * k + 1];
            }
            cand.push_back(d);
        }

        // ---- 逐类 NMS ----
        for (int c = 0; c < NUM_CLS; ++c) {
            vector<Rect>  boxes;
            vector<float> scores;
            vector<int>   srcIdx;

            for (size_t i = 0; i < cand.size(); ++i) {
                if (cand[i].cls != c) continue;

                int x = (int)std::lround(cand[i].box.x);
                int y = (int)std::lround(cand[i].box.y);
                int w = (int)std::lround(cand[i].box.width);
                int h = (int)std::lround(cand[i].box.height);
                if (w < 1 || h < 1) continue;

                boxes.push_back(Rect(x, y, w, h));
                scores.push_back(cand[i].score);
                srcIdx.push_back((int)i);
            }
            if (boxes.empty()) continue;

            vector<int> keep;
            dnn::NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES, keep);
            for (size_t k = 0; k < keep.size(); ++k)
                dets.push_back(cand[srcIdx[keep[k]]]);
        }

        // ---- letterbox 反变换：640 坐标 -> 原图坐标 ----
        const float sc = (lb.scale > 1e-6f) ? lb.scale : 1.0f;
        const float W  = (float)bgr.cols;
        const float H  = (float)bgr.rows;

        for (size_t i = 0; i < dets.size(); ++i) {
            Detection& d = dets[i];

            float x1 = (d.box.x - lb.padX) / sc;
            float y1 = (d.box.y - lb.padY) / sc;
            float x2 = (d.box.x + d.box.width  - lb.padX) / sc;
            float y2 = (d.box.y + d.box.height - lb.padY) / sc;

            x1 = std::max(0.0f, std::min(x1, W - 1.0f));
            y1 = std::max(0.0f, std::min(y1, H - 1.0f));
            x2 = std::max(0.0f, std::min(x2, W - 1.0f));
            y2 = std::max(0.0f, std::min(y2, H - 1.0f));
            d.box = Rect2f(Point2f(x1, y1), Point2f(x2, y2));

            for (int k = 0; k < NUM_KPT; ++k) {
                float kx = (d.kpts[k].x - lb.padX) / sc;
                float ky = (d.kpts[k].y - lb.padY) / sc;
                d.kpts[k].x = std::max(0.0f, std::min(kx, W - 1.0f));
                d.kpts[k].y = std::max(0.0f, std::min(ky, H - 1.0f));
            }
        }

        auto tick3 = chrono::steady_clock::now();
        msPost = chrono::duration<double, std::milli>(tick3 - tick2).count();
    }

private:
    dnn::Net net_;
};

// ---------------------------------------------------------------------------
//  画图
// ---------------------------------------------------------------------------
// 采样框内"亮且有颜色"的像素，算出目标的主色调，返回 BGR。
// 采样不到（太暗/太灰）返回 Scalar(-1,-1,-1) 当哨兵。
static Scalar sampleObjectColor(const Mat& img, const Rect& box, int step = 2)
{
    const Rect r = box & Rect(0, 0, img.cols, img.rows);
    if (r.width < 4 || r.height < 4) return Scalar(-1, -1, -1);

    Mat hsv;
    cvtColor(img(r), hsv, COLOR_BGR2HSV);

    double sx = 0.0, sy = 0.0;      // 色相的环形累加（H 是 0~180 的环）
    long n = 0, sumS = 0, sumV = 0;

    for (int y = 0; y < hsv.rows; y += step) {
        const Vec3b* p = hsv.ptr<Vec3b>(y);
        for (int x = 0; x < hsv.cols; x += step) {
            const int v = p[x][2];
            const int s = p[x][1];
            if (v < 90 || s < 70) continue;          // 太暗或太灰，当背景丢掉
            const double a = p[x][0] * CV_PI / 90.0;
            sx += std::cos(a);
            sy += std::sin(a);
            sumS += s;
            sumV += v;
            ++n;
        }
    }
    if (n < 20) return Scalar(-1, -1, -1);

    double hue = std::atan2(sy, sx) * 90.0 / CV_PI;
    if (hue < 0.0) hue += 180.0;

    const int s = (int)std::min<long>(255, std::max<long>(140, sumS / n));
    const int v = (int)std::min<long>(255, std::max<long>(160, sumV / n));

    Mat one(1, 1, CV_8UC3, Scalar(hue, s, v));
    Mat bgr;
    cvtColor(one, bgr, COLOR_HSV2BGR);
    const Vec3b c = bgr.at<Vec3b>(0, 0);
    return Scalar(c[0], c[1], c[2]);
}

static void drawResult(Mat& img, const vector<Detection>& dets)
{
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];

        const int cidx = std::max(0, std::min(d.cls, NUM_CLS - 1));
        const char*  cname = CLASS_NAMES[cidx];

        Rect box((int)std::lround(d.box.x), (int)std::lround(d.box.y),
                 (int)std::lround(d.box.width), (int)std::lround(d.box.height));

        // 颜色：默认按目标本身的颜色画，采样失败才退回类别色
        Scalar color = CLASS_COLORS[cidx];
        if (USE_OBJECT_COLOR) {
            const Scalar obj = sampleObjectColor(img, box);   // 必须在画框之前采样
            if (obj[0] >= 0.0) color = obj;
        }

        rectangle(img, box, color, 2);

        // 4 个关键点连成四边形
        vector<Point> poly;
        poly.reserve(d.kpts.size());
        for (size_t k = 0; k < d.kpts.size(); ++k)
            poly.push_back(Point((int)std::lround(d.kpts[k].x), (int)std::lround(d.kpts[k].y)));

        if (poly.size() >= 3) polylines(img, poly, true, color, 1, LINE_AA);
        for (size_t k = 0; k < poly.size(); ++k)
            circle(img, poly[k], 4, color, -1, LINE_AA);

        // 标签
        char text[128];
        snprintf(text, sizeof(text), "%s %.2f", cname, d.score);

        int baseline = 0;
        Size ts = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int ty = std::max(box.y - 6, ts.height + 6);
        rectangle(img, Rect(box.x, ty - ts.height - 4, ts.width + 8, ts.height + baseline + 6),
                  color, FILLED);
        putText(img, text, Point(box.x + 4, ty),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, LINE_AA);

        // ---- 检测框中心点（作业进阶要求）----
        const Point ctr((int)std::lround(d.box.x + d.box.width  * 0.5f),
                        (int)std::lround(d.box.y + d.box.height * 0.5f));

        const Scalar cCol(0, 255, 255);          // 黄色准星
        const int L = 12;
        line(img, Point(ctr.x - L, ctr.y), Point(ctr.x + L, ctr.y), cCol, 2, LINE_AA);
        line(img, Point(ctr.x, ctr.y - L), Point(ctr.x, ctr.y + L), cCol, 2, LINE_AA);
        circle(img, ctr, 3, Scalar(0, 0, 0), FILLED, LINE_AA);

        char ctext[64];
        snprintf(ctext, sizeof(ctext), "C(%d,%d)", ctr.x, ctr.y);
        const int cy = std::min(box.y + box.height + 20, img.rows - 8);
        putText(img, ctext, Point(box.x, cy),
                FONT_HERSHEY_SIMPLEX, 0.55, cCol, 2, LINE_AA);
    }
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // printf 在 Qt Creator / 重定向到管道时是全缓冲的，日志会憋很久才吐。
    // 关掉缓冲，每条日志立刻出现在「应用程序输出」里。
    setvbuf(stdout, nullptr, _IONBF, 0);

    // 参数：
    //   --imgsz N      模型输入尺寸（不填则自动探测）
    //   --exposure N   曝光时间 µs（对照实验用）
    //   --gain N       增益 dB（对照实验用）
    //   其余任意一个参数当视频路径
    string videoPath;
    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--imgsz" && i + 1 < argc) {
            const int v = atoi(argv[++i]);
            if (v >= 64 && v % 32 == 0) {
                INPUT_W = INPUT_H = v;
            } else {
                printf("[参数] --imgsz 必须是 >=64 的 32 的倍数，已忽略\n");
            }
        } else if (a == "--exposure" && i + 1 < argc) {
            g_exposureUs = atoi(argv[++i]);
        } else if (a == "--gain" && i + 1 < argc) {
            g_gain = (float)atof(argv[++i]);
        } else {
            videoPath = a;
        }
    }

    printf("===== hik_yolo_pose =====\n");
    printf("[MAIN] 模型输入尺寸: %d x %d\n", INPUT_W, INPUT_H);
    printf("[MAIN] 曝光 %d us | 增益 %.1f dB\n", g_exposureUs, g_gain);

    // 让 OpenCV 用满所有 CPU 核（虚拟机里对 FPS 有帮助）
    setNumThreads((int)std::thread::hardware_concurrency());

    PoseModel model;
    if (!model.load(MODEL_PATH)) return -1;

    HikCamera cam;
    VideoCapture cap;
    const bool useCamera = videoPath.empty();

    if (useCamera) {
        if (!cam.open(0)) {
            printf("[MAIN] 相机打开失败\n");
            return -1;
        }
        printf("[MAIN] 相机已启动，开始取流\n");
    } else {
        cap.open(videoPath);
        if (!cap.isOpened()) {
            printf("[MAIN] 打不开视频: %s\n", videoPath.c_str());
            return -1;
        }
        printf("[MAIN] 使用视频源: %s\n", videoPath.c_str());
    }

    // WINDOW_AUTOSIZE：imshow 不再每帧缩放图像，省 3~8 ms
    // 想手动缩放窗口就换回 WINDOW_NORMAL + resizeWindow
    namedWindow(WIN_NAME, WINDOW_AUTOSIZE);

    Fps fps;
    int frameCount = 0;

    // 分阶段计时的累加器（每 30 帧打印一次平均值）
    double accPre = 0.0, accInfer = 0.0, accPost = 0.0, accUi = 0.0;
    double accConv = 0.0, accE2E = 0.0;

    // CPU 占用统计（进程口径）
    double cpuLast = (double)std::clock() / (double)CLOCKS_PER_SEC;
    auto   cpuWall0 = chrono::steady_clock::now();

    while (true) {
        Mat frame;

        if (useCamera) {
            if (!cam.grab(frame)) {
                // 超时或断流：别空转，顺手刷一下窗口
                if ((waitKey(1) & 0xFF) == 27) break;
                continue;
            }
        } else {
            cap >> frame;
            if (frame.empty()) break;
        }

        vector<Detection> dets;

        // 端到端耗时：从"拿到图像"到"推理结果出来"
        const auto tE2E0 = chrono::steady_clock::now();
        model.infer(frame, dets);
        const double e2eMs = chrono::duration<double, std::milli>(
                                 chrono::steady_clock::now() - tE2E0).count();

        auto uiStart = chrono::steady_clock::now();

        drawResult(frame, dets);

        ++frameCount;
        const double fpsNow = fps.tick();      // 每帧只取一次，避免 HUD 和日志各算一遍
        char hud[128];
        snprintf(hud, sizeof(hud), "FPS %.1f | objects %d",
                 fpsNow, (int)dets.size());
        putText(frame, hud, Point(20, 40), FONT_HERSHEY_SIMPLEX,
                0.9, Scalar(0, 255, 0), 2, LINE_AA);

        imshow(WIN_NAME, frame);

        int key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        const double uiMs =
            chrono::duration<double, std::milli>(chrono::steady_clock::now() - uiStart).count();

        accPre   += model.msPre;
        accInfer += model.msInfer;
        accPost  += model.msPost;
        accUi    += uiMs;
        accConv  += cam.lastConvertMs;
        accE2E   += e2eMs;

        if (frameCount % 30 == 0) {
            const double k = 1.0 / 30.0;
            printf("[time] pre %6.1f | infer %7.1f | post %5.1f | ui %5.1f | total %7.1f ms\n",
                   accPre * k, accInfer * k, accPost * k, accUi * k,
                   (accPre + accInfer + accPost + accUi) * k);

            // CPU 占用（进程口径，多核可超过 100%）
            const double cpuNow = (double)std::clock() / (double)CLOCKS_PER_SEC;
            const double wall   = chrono::duration<double>(chrono::steady_clock::now() - cpuWall0).count();
            const double cpuPct = (wall > 1e-6) ? (cpuNow - cpuLast) / wall * 100.0 : 0.0;
            cpuLast = cpuNow;
            cpuWall0 = chrono::steady_clock::now();

            printf("[stat] fps %.1f | cpu %.0f%% | convert %.2f ms | e2e %.1f ms | %dx%d\n",
                   fpsNow, cpuPct, accConv * k, accE2E * k,
                   cam.lastWidth, cam.lastHeight);

            accPre = accInfer = accPost = accUi = 0.0;
            accConv = accE2E = 0.0;

            // ---- 检测框中心点坐标（作业进阶要求：终端输出）----
            if (dets.empty()) {
                printf("[center] 无目标\n");
            } else {
                for (size_t k2 = 0; k2 < dets.size(); ++k2) {
                    const Detection& d = dets[k2];
                    const int cx = (int)std::lround(d.box.x + d.box.width  * 0.5f);
                    const int cy = (int)std::lround(d.box.y + d.box.height * 0.5f);
                    printf("[center] obj%zu  cls=%s  conf=%.2f  center=(%d, %d)  "
                           "box=(%d,%d,%dx%d)\n",
                           k2 + 1,
                           CLASS_NAMES[std::max(0, std::min(d.cls, NUM_CLS - 1))],
                           d.score, cx, cy,
                           (int)std::lround(d.box.x), (int)std::lround(d.box.y),
                           (int)std::lround(d.box.width), (int)std::lround(d.box.height));
                }
            }
        }
    }

    cam.close();
    if (cap.isOpened()) cap.release();
    destroyAllWindows();

    printf("[MAIN] 安全退出，共处理 %d 帧\n", frameCount);
    return 0;
}
