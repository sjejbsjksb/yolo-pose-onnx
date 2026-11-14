// ============================================================================
//  main.cpp —— 第五次作业：装甲板识别 + PnP 位姿解算 + 卡尔曼滤波
//
//  流程：相机采集 → YOLO 识别 → 角点提取 → solvePnP → 卡尔曼滤波 → 运动预测
//
//  依赖：
//      best.onnx     装甲板检测模型（第四次作业的 6 类 + 4 关键点模型）
//      camera.yaml   相机标定结果（由 calibrate.cpp 生成）
//
//  编译：见 CMakeLists.txt
//  运行：sudo ./ArmorPnP
//        sudo ./ArmorPnP --exposure 10000 --gain 0
//
//  注意：
//      1. camera.yaml 的内参必须与运行时分辨率一致（本程序不裁剪 ROI）
//      2. 装甲板实际尺寸在下面的 ARMOR_W_MM / ARMOR_H_MM 里改
//      3. 关键点顺序必须与三维物点顺序一致，详见 KPT_ORDER 说明
// ============================================================================

#include <MvCameraControl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
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
//  配置区
// ---------------------------------------------------------------------------
static const char* MODEL_PATH  = "best.onnx";
static const char* CALIB_YAML  = "camera.yaml";
static const char* CSV_PATH    = "pose_log.csv";

static int INPUT_W = 640;
static int INPUT_H = 640;

static const int   NUM_CLS  = 6;
static const int   NUM_KPT  = 4;
static const int   KPT_DIMS = 2;

static const float CONF_THRES = 0.50f;
static const float NMS_THRES  = 0.40f;

// ---- 装甲板真实尺寸（毫米）—— 实测值 ----
//   装甲板整体: 长 224  x  宽 126
//   灯带长度  : 54
// 说明：这里的宽高指的是「模型 4 个关键点所围住的那块矩形」。
//       若你的关键点标的是灯条范围而不是整块板，要改成灯条的实际尺寸。
// 自检方法：把装甲板放在 1 米处，看 Z 是否约等于 1000mm；按比例修正即可。
static const float ARMOR_W_MM   = 224.0f;   // 宽
static const float ARMOR_H_MM   = 126.0f;   // 高
static const float LED_STRIP_MM = 54.0f;    // 灯带长度（备用）

// ---- 角点来源 ----
// true  : 用模型输出的 4 个关键点（推荐，测的就是四个角）
// false : 用检测框的四个角（关键点不稳时的退路）
static const bool USE_KEYPOINTS = true;

// ---- 卡尔曼参数 ----
static const float KF_MEAS_NOISE  = 4.0f;    // 测量噪声（mm²），越大越信模型
static const float KF_PROC_NOISE  = 0.8f;    // 过程噪声，越大越信测量
static const float PREDICT_HORIZON_S = 0.15f; // 预测提前量（秒）

static int   g_exposureUs = 5000;
static float g_gain       = 0.0f;
static string g_pixelFmt  = "bgr";

static const char* WIN_NAME = "Armor PnP + Kalman";

static const Scalar CLASS_COLORS[] = {
    Scalar(0,   0, 255), Scalar(255, 0,   0), Scalar(0, 255,   0),
    Scalar(0, 255, 255), Scalar(255, 0, 255), Scalar(255, 255, 0),
};

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
    float scale = 1.f;
    float padX  = 0.f;
    float padY  = 0.f;
};

// ---------------------------------------------------------------------------
//  小工具
// ---------------------------------------------------------------------------
static void mvsError(const char* api, int code)
{
    printf("[MVS] %s 失败, code = 0x%X\n", api, (unsigned int)code);
}

static string camString(void* h, const char* key)
{
    MVCC_STRINGVALUE sv;
    memset(&sv, 0, sizeof(sv));
    sv.nMaxLength = (int64_t)sizeof(sv.chCurValue);
    if (MV_CC_GetStringValue(h, key, &sv) != MV_OK) return string("N/A");
    return string(sv.chCurValue);
}

static Mat letterbox(const Mat& src, int newW, int newH, LetterboxInfo& info)
{
    const int w = src.cols, h = src.rows;
    const float r = std::min((float)newW / (float)w, (float)newH / (float)h);
    const int nw = (int)std::lround(w * r);
    const int nh = (int)std::lround(h * r);

    Mat resized;
    resize(src, resized, Size(nw, nh));

    const int dw = newW - nw, dh = newH - nh;
    const int top = dh / 2, bottom = dh - top;
    const int left = dw / 2, right = dw - left;

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
//  卡尔曼滤波：状态 [x, y, z, vx, vy, vz]，观测 [x, y, z]（单位 mm）
// ---------------------------------------------------------------------------
class Kalman3D
{
public:
    void init()
    {
        kf_.init(6, 3, 0, CV_32F);

        // 观测矩阵：只测位置
        kf_.measurementMatrix = Mat::zeros(3, 6, CV_32F);
        kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
        kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
        kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

        // 状态转移：位置 += 速度 * dt（dt 每帧更新）
        kf_.transitionMatrix = Mat::eye(6, 6, CV_32F);

        kf_.processNoiseCov     = Mat::eye(6, 6, CV_32F) * KF_PROC_NOISE;
        kf_.measurementNoiseCov = Mat::eye(3, 3, CV_32F) * KF_MEAS_NOISE;
        kf_.errorCovPost        = Mat::eye(6, 6, CV_32F) * 100.0f;

        inited_ = false;
    }

    bool inited() const { return inited_; }

    void predictOnly(float dt) { setDt(dt); kf_.predict(); }    // 有测量时调用：predict + correct
    void update(float dt, const Point3f& meas)
    {
        setDt(dt);
        kf_.predict();

        Mat m(3, 1, CV_32F);
        m.at<float>(0, 0) = meas.x;
        m.at<float>(1, 0) = meas.y;
        m.at<float>(2, 0) = meas.z;

        if (!inited_) {
            // 第一帧：直接把测量值写进状态，速度置 0
            kf_.statePost = Mat::zeros(6, 1, CV_32F);
            kf_.statePost.at<float>(0, 0) = meas.x;
            kf_.statePost.at<float>(1, 0) = meas.y;
            kf_.statePost.at<float>(2, 0) = meas.z;
            inited_ = true;
            return;
        }
        kf_.correct(m);
    }

    Point3f pos() const
    {
        const Mat& s = kf_.statePost;
        return Point3f(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0));
    }

    Point3f vel() const
    {
        const Mat& s = kf_.statePost;
        return Point3f(s.at<float>(3, 0), s.at<float>(4, 0), s.at<float>(5, 0));
    }

    // 预测 horizon 秒后的位置
    Point3f predictAhead(float horizon) const
    {
        const Point3f p = pos(), v = vel();
        return Point3f(p.x + v.x * horizon, p.y + v.y * horizon, p.z + v.z * horizon);
    }

private:
    void setDt(float dt)
    {
        kf_.transitionMatrix.at<float>(0, 3) = dt;
        kf_.transitionMatrix.at<float>(1, 4) = dt;
        kf_.transitionMatrix.at<float>(2, 5) = dt;
    }

    KalmanFilter kf_;
    bool        inited_ = false;
};

// ---------------------------------------------------------------------------
//  海康相机
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
        if (ret != MV_OK) { mvsError("MV_CC_EnumDevices", ret); return false; }
        if (list.nDeviceNum == 0) {
            printf("[MVS] 一个相机都没枚举到。检查连接、是否 sudo、udev 规则。\n");
            return false;
        }
        printf("[MVS] 枚举到 %u 个设备\n", list.nDeviceNum);
        if (index < 0 || (unsigned int)index >= list.nDeviceNum) {
            printf("[MVS] 相机序号越界: %d\n", index);
            return false;
        }

        ret = MV_CC_CreateHandle(&handle_, list.pDeviceInfo[index]);
        if (ret != MV_OK) { mvsError("MV_CC_CreateHandle", ret); handle_ = nullptr; return false; }

        ret = MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0);
        if (ret != MV_OK) {
            mvsError("MV_CC_OpenDevice", ret);
            if ((unsigned int)ret == (unsigned int)MV_E_ACCESS_DENIED)
                printf("[MVS] 权限不足：用 sudo 运行。\n");
            close();
            return false;
        }

        printf("[INFO] 相机型号 : %s\n", camString(handle_, "DeviceModelName").c_str());
        printf("[INFO] 序列号   : %s\n", camString(handle_, "DeviceSerialNumber").c_str());
        printf("[INFO] 固件版本 : %s\n", camString(handle_, "DeviceVersion").c_str());

        MV_CC_SetEnumValue(handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);
        MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);
        MV_CC_SetFloatValue(handle_, "ExposureTime", (float)g_exposureUs);
        MV_CC_SetFloatValue(handle_, "Gain", g_gain);

        unsigned int pf = PixelType_Gvsp_BGR8_Packed;
        if (g_pixelFmt == "rgb")        pf = PixelType_Gvsp_RGB8_Packed;
        else if (g_pixelFmt == "bayer") pf = PixelType_Gvsp_BayerRG8;
        else if (g_pixelFmt == "mono")  pf = PixelType_Gvsp_Mono8;
        if (MV_CC_SetEnumValue(handle_, "PixelFormat", pf) != MV_OK)
            MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);

        const int pkt = MV_CC_GetOptimalPacketSize(handle_);
        if (pkt > 0) MV_CC_SetIntValue(handle_, "GevSCPSPacketSize", (unsigned)pkt);

        ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) { mvsError("MV_CC_StartGrabbing", ret); close(); return false; }
        grabbing_ = true;
        return true;
    }

    bool grab(Mat& bgr, int timeoutMs = 1000)
    {
        if (!handle_) return false;

        MV_FRAME_OUT fo;
        memset(&fo, 0, sizeof(fo));
        if (MV_CC_GetImageBuffer(handle_, &fo, (unsigned)timeoutMs) != MV_OK) return false;

        const int w = (int)fo.stFrameInfo.nWidth;
        const int h = (int)fo.stFrameInfo.nHeight;
        if (w <= 0 || h <= 0) { MV_CC_FreeImageBuffer(handle_, &fo); return false; }

        lastWidth = w; lastHeight = h;
        const size_t need = (size_t)w * h * 3u;
        if (buf_.size() < need) buf_.resize(need);

        if (fo.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed) {
            memcpy(buf_.data(), fo.pBufAddr, need);
            MV_CC_FreeImageBuffer(handle_, &fo);
        } else {
            MV_CC_PIXEL_CONVERT_PARAM cvt;
            memset(&cvt, 0, sizeof(cvt));
            cvt.nWidth = (unsigned short)w;
            cvt.nHeight = (unsigned short)h;
            cvt.pSrcData = fo.pBufAddr;
            cvt.nSrcDataLen = fo.stFrameInfo.nFrameLen;
            cvt.enSrcPixelType = fo.stFrameInfo.enPixelType;
            cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
            cvt.pDstBuffer = buf_.data();
            cvt.nDstBufferSize = (unsigned)need;
            const int ret = MV_CC_ConvertPixelType(handle_, &cvt);
            MV_CC_FreeImageBuffer(handle_, &fo);
            if (ret != MV_OK) return false;
        }
        bgr = Mat(h, w, CV_8UC3, buf_.data());
        return true;
    }

    void close()
    {
        if (!handle_) return;
        if (grabbing_) { MV_CC_StopGrabbing(handle_); grabbing_ = false; }
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }

    int lastWidth = 0, lastHeight = 0;

private:
    void*                 handle_   = nullptr;
    bool                  grabbing_ = false;
    vector<unsigned char> buf_;
};

// ---------------------------------------------------------------------------
//  检测模型
// ---------------------------------------------------------------------------
class PoseModel
{
public:
    bool load(const char* path)
    {
        try { net_ = dnn::readNetFromONNX(path); }
        catch (const cv::Exception& e) {
            printf("[YOLO] 加载 ONNX 失败: %s\n", e.what());
            return false;
        }
        if (net_.empty()) { printf("[YOLO] 网络为空: %s\n", path); return false; }

        net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
        net_.enableFusion(true);

        vector<int> sizes;
        sizes.push_back(INPUT_W);
        const int common[] = { 640, 480, 416, 320 };
        for (int k = 0; k < 4; ++k) {
            const int s = common[k];
            if (std::find(sizes.begin(), sizes.end(), s) == sizes.end()) sizes.push_back(s);
        }

        const utils::logging::LogLevel oldLevel =
            utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

        for (size_t i = 0; i < sizes.size(); ++i) {
            const int s = sizes[i];
            if (i > 0) {
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
                printf("[YOLO] 模型输入尺寸: %d x %d%s\n", s, s, (i == 0 ? "" : "（自动探测）"));
                return true;
            } catch (const cv::Exception&) { /* 换下一个尺寸 */ }
        }
        utils::logging::setLogLevel(oldLevel);
        printf("[YOLO] 640/480/416/320 都跑不通，模型可能没打补丁\n");
        return false;
    }

    void infer(const Mat& bgr, vector<Detection>& dets)
    {
        dets.clear();
        if (bgr.empty()) return;

        LetterboxInfo lb;
        Mat lbImg = letterbox(bgr, INPUT_W, INPUT_H, lb);

        Mat blob;
        dnn::blobFromImage(lbImg, blob, 1.0 / 255.0, Size(INPUT_W, INPUT_H),
                           Scalar(), true, false, CV_32F);
        net_.setInput(blob);

        Mat out;
        try { out = net_.forward(); }
        catch (const cv::Exception& e) {
            printf("[YOLO] forward 失败: %s\n", e.what());
            return;
        }

        if (out.dims != 3) return;
        const int C = out.size[1];
        const int N = out.size[2];
        if (C != 4 + NUM_CLS + KPT_DIMS * NUM_KPT) return;

        Mat m2d = out.reshape(1, C);
        Mat rows;
        transpose(m2d, rows);

        vector<Detection> cand;
        for (int i = 0; i < N; ++i) {
            const float* r = rows.ptr<float>(i);

            float score = -1.f; int cls = -1;
            for (int c = 0; c < NUM_CLS; ++c)
                if (r[4 + c] > score) { score = r[4 + c]; cls = c; }
            if (score < CONF_THRES) continue;

            const float cx = r[0], cy = r[1], bw = r[2], bh = r[3];
            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(bw) || !std::isfinite(bh)) continue;
            if (bw < 1.f || bh < 1.f) continue;

            Detection d;
            d.score = score;
            d.cls = cls;
            d.box = Rect2f(cx - 0.5f * bw, cy - 0.5f * bh, bw, bh);
            d.kpts.resize(NUM_KPT);
            for (int k = 0; k < NUM_KPT; ++k) {
                d.kpts[k].x = r[4 + NUM_CLS + KPT_DIMS * k + 0];
                d.kpts[k].y = r[4 + NUM_CLS + KPT_DIMS * k + 1];
            }
            cand.push_back(d);
        }

        for (int c = 0; c < NUM_CLS; ++c) {
            vector<Rect> boxes; vector<float> scores; vector<int> srcIdx;
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
            for (size_t k = 0; k < keep.size(); ++k) dets.push_back(cand[srcIdx[keep[k]]]);
        }

        const float sc = (lb.scale > 1e-6f) ? lb.scale : 1.f;
        const float W = (float)bgr.cols, H = (float)bgr.rows;

        for (size_t i = 0; i < dets.size(); ++i) {
            Detection& d = dets[i];
            float x1 = (d.box.x - lb.padX) / sc;
            float y1 = (d.box.y - lb.padY) / sc;
            float x2 = (d.box.x + d.box.width - lb.padX) / sc;
            float y2 = (d.box.y + d.box.height - lb.padY) / sc;
            x1 = std::max(0.f, std::min(x1, W - 1.f));
            y1 = std::max(0.f, std::min(y1, H - 1.f));
            x2 = std::max(0.f, std::min(x2, W - 1.f));
            y2 = std::max(0.f, std::min(y2, H - 1.f));
            d.box = Rect2f(Point2f(x1, y1), Point2f(x2, y2));

            for (int k = 0; k < NUM_KPT; ++k) {
                d.kpts[k].x = std::max(0.f, std::min((d.kpts[k].x - lb.padX) / sc, W - 1.f));
                d.kpts[k].y = std::max(0.f, std::min((d.kpts[k].y - lb.padY) / sc, H - 1.f));
            }
        }
    }

private:
    dnn::Net net_;
};

// ---------------------------------------------------------------------------
//  PnP 位姿解算
// ---------------------------------------------------------------------------
// 三维物点：装甲板中心为原点，z=0 平面，单位为 mm
// 顺序必须和图像角点一一对应：
//     0: 左上(-w/2, -h/2)  1: 右上(w/2, -h/2)
//     2: 右下(w/2,  h/2)   3: 左下(-w/2, h/2)
static vector<Point3f> makeArmorObjectPoints()
{
    const float hw = ARMOR_W_MM * 0.5f;
    const float hh = ARMOR_H_MM * 0.5f;
    return {
        Point3f(-hw, -hh, 0.f),
        Point3f( hw, -hh, 0.f),
        Point3f( hw,  hh, 0.f),
        Point3f(-hw,  hh, 0.f),
    };
}

static bool solveArmorPose(const vector<Point2f>& imgPts,
                           const vector<Point3f>& objPts,
                           const Mat& K, const Mat& D,
                           Mat& rvec, Mat& tvec)
{
    if (imgPts.size() != objPts.size() || imgPts.size() < 4) return false;
    try {
        solvePnP(objPts, imgPts, K, D, rvec, tvec,
                 false, SOLVEPNP_IPPE);        // IPPE 专用于平面四点
        solvePnPRefineLM(objPts, imgPts, K, D, rvec, tvec);
    } catch (const cv::Exception& e) {
        printf("[PnP] 解算失败: %s\n", e.what());
        return false;
    }
    return true;
}

// 画坐标轴：X 红 / Y 绿 / Z 蓝，长度 axisLenMm
static void drawAxes(Mat& img, const Mat& K, const Mat& D,
                     const Mat& rvec, const Mat& tvec, float axisLenMm)
{
    vector<Point3f> axis = {
        Point3f(0, 0, 0),
        Point3f(axisLenMm, 0, 0),
        Point3f(0, axisLenMm, 0),
        Point3f(0, 0, axisLenMm),
    };
    vector<Point2f> proj;
    projectPoints(axis, rvec, tvec, K, D, proj);

    line(img, proj[0], proj[1], Scalar(0, 0, 255), 3, LINE_AA);
    line(img, proj[0], proj[2], Scalar(0, 255, 0), 3, LINE_AA);
    line(img, proj[0], proj[3], Scalar(255, 0, 0), 3, LINE_AA);
}

// ---------------------------------------------------------------------------
//  主程序
// ---------------------------------------------------------------------------
static volatile sig_atomic_t g_interrupted = 0;
static void onSigInt(int) { g_interrupted = 1; }

static void drawHud(Mat& img, const vector<Detection>& dets, double fps,
                    bool hasPose, bool hasRaw,
                    const Point3f& raw, const Point3f& filt, const Point3f& pred)
{
    char buf[256];

    // 检测框 + 关键点
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        const Scalar color = CLASS_COLORS[d.cls % 6];
        Rect box((int)std::lround(d.box.x), (int)std::lround(d.box.y),
                 (int)std::lround(d.box.width), (int)std::lround(d.box.height));
        rectangle(img, box, color, 2);

        vector<Point> poly;
        for (size_t k = 0; k < d.kpts.size(); ++k)
            poly.push_back(Point((int)d.kpts[k].x, (int)d.kpts[k].y));
        if (poly.size() >= 3) polylines(img, poly, true, color, 1, LINE_AA);
        for (size_t k = 0; k < poly.size(); ++k) {
            circle(img, poly[k], 4, color, -1, LINE_AA);
            char idx[16]; snprintf(idx, sizeof(idx), "%zu", k);
            putText(img, idx, poly[k] + Point(6, -6), FONT_HERSHEY_SIMPLEX, 0.45, color, 1, LINE_AA);
        }
        snprintf(buf, sizeof(buf), "cls%d %.2f", d.cls, d.score);
        putText(img, buf, Point(box.x, std::max(box.y - 8, 16)),
                FONT_HERSHEY_SIMPLEX, 0.55, color, 2, LINE_AA);
    }

    // 左上角 FPS
    snprintf(buf, sizeof(buf), "FPS %.1f | objects %d", fps, (int)dets.size());
    putText(img, buf, Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, LINE_AA);

    // 左下角位姿信息
    if (hasPose) {
        int y = img.rows - 110;

        if (hasRaw) {
            const float dist = std::sqrt(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z);
            snprintf(buf, sizeof(buf), "X %7.1f  Y %7.1f  Z %7.1f  mm", raw.x, raw.y, raw.z);
            putText(img, buf, Point(20, y), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2, LINE_AA);
            snprintf(buf, sizeof(buf), "dist %.1f mm (%.2f m)", dist, dist / 1000.0f);
            putText(img, buf, Point(20, y + 26), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2, LINE_AA);

            snprintf(buf, sizeof(buf), "raw  (%.1f, %.1f, %.1f)", raw.x, raw.y, raw.z);
            putText(img, buf, Point(20, y + 56), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(180, 180, 180), 1, LINE_AA);
        } else {
            putText(img, "raw  (---)  target lost, showing prediction",
                    Point(20, y), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 255), 2, LINE_AA);
        }

        snprintf(buf, sizeof(buf), "filt (%.1f, %.1f, %.1f)", filt.x, filt.y, filt.z);
        putText(img, buf, Point(20, y + 78), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);
        snprintf(buf, sizeof(buf), "pred (%.1f, %.1f, %.1f)", pred.x, pred.y, pred.z);
        putText(img, buf, Point(20, y + 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1, LINE_AA);
    } else {
        putText(img, "no pose (camera.yaml missing or no detection)",
                Point(20, img.rows - 30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 255), 2, LINE_AA);
    }
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    signal(SIGINT, onSigInt);

    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--imgsz" && i + 1 < argc) {
            const int v = atoi(argv[++i]);
            if (v >= 64 && v % 32 == 0) INPUT_W = INPUT_H = v;
        } else if (a == "--exposure" && i + 1 < argc) {
            g_exposureUs = atoi(argv[++i]);
        } else if (a == "--gain" && i + 1 < argc) {
            g_gain = (float)atof(argv[++i]);
        } else if (a == "--pixel" && i + 1 < argc) {
            g_pixelFmt = argv[++i];
        }
    }

    printf("===== Armor PnP + Kalman =====\n");
    printf("[MAIN] 装甲板尺寸: %.1f x %.1f mm\n", ARMOR_W_MM, ARMOR_H_MM);

    // ---- 标定参数 ----
    Mat K, D;
    bool hasCalib = false;
    {
        FileStorage fs(CALIB_YAML, FileStorage::READ);
        if (fs.isOpened()) {
            fs["camera_matrix"] >> K;
            fs["distortion_coefficients"] >> D;
            if (!K.empty() && !D.empty()) {
                hasCalib = true;
                int cw = 0, ch = 0;
                fs["image_width"] >> cw;
                fs["image_height"] >> ch;
                double rms = 0; fs["rms"] >> rms;
                printf("[CAL] 已加载 %s | 标定分辨率 %dx%d | RMS %.3f px\n",
                       CALIB_YAML, cw, ch, rms);
                printf("[CAL] fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
                       K.at<double>(0, 0), K.at<double>(1, 1),
                       K.at<double>(0, 2), K.at<double>(1, 2));
            }
        }
        if (!hasCalib)
            printf("[CAL] 没找到 %s —— 只显示检测框，不做位姿解算。\n"
                   "      先运行 ./Calibrate 采集棋盘格并标定。\n", CALIB_YAML);
    }

    const vector<Point3f> objPts = makeArmorObjectPoints();

    PoseModel model;
    if (!model.load(MODEL_PATH)) return -1;

    HikCamera cam;
    if (!cam.open(0)) return -1;

    namedWindow(WIN_NAME, WINDOW_AUTOSIZE);

    Kalman3D kalman;
    kalman.init();

    FILE* csv = fopen(CSV_PATH, "w");
    if (csv) {
        fprintf(csv, "t_ms,detected,raw_x,raw_y,raw_z,filt_x,filt_y,filt_z,pred_x,pred_y,pred_z\n");
        printf("[LOG] 位姿数据写入 %s\n", CSV_PATH);
    }

    auto tStart = chrono::steady_clock::now();
    auto tLast  = tStart;

    Fps fps;
    int frameCount = 0;

    while (!g_interrupted) {
        Mat frame;
        if (!cam.grab(frame)) {
            if ((waitKey(1) & 0xFF) == 27) break;
            continue;
        }

        vector<Detection> dets;
        model.infer(frame, dets);

        // ---- 选置信度最高的目标做位姿解算 ----
        bool poseOk = false;
        Mat  rvec, tvec;
        Point3f raw(0, 0, 0);

        if (hasCalib && !dets.empty()) {
            int best = 0;
            for (size_t i = 1; i < dets.size(); ++i)
                if (dets[i].score > dets[best].score) best = (int)i;

            vector<Point2f> corners;
            if (USE_KEYPOINTS) {
                corners = dets[best].kpts;
            } else {
                const Rect2f& b = dets[best].box;
                corners = { Point2f(b.x, b.y),
                            Point2f(b.x + b.width, b.y),
                            Point2f(b.x + b.width, b.y + b.height),
                            Point2f(b.x, b.y + b.height) };
            }

            if (corners.size() == 4 && solveArmorPose(corners, objPts, K, D, rvec, tvec)) {
                raw = Point3f((float)tvec.at<double>(0),
                              (float)tvec.at<double>(1),
                              (float)tvec.at<double>(2));
                poseOk = true;
            }
        }

        // ---- 卡尔曼：每帧都预测，有测量就更新 ----
        const auto now = chrono::steady_clock::now();
        float dt = (float)chrono::duration<double>(now - tLast).count();
        tLast = now;
        if (dt <= 0.f || dt > 1.f) dt = 0.1f;      // 异常值兜底

        if (poseOk) kalman.update(dt, raw);
        else if (kalman.inited()) kalman.predictOnly(dt);

        const Point3f filt = kalman.pos();
        const Point3f pred = kalman.predictAhead(PREDICT_HORIZON_S);
        const double fpsNow = fps.tick();          // 每帧只取一次

        // ---- 绘制 ----
        drawHud(frame, dets, fpsNow, kalman.inited(), poseOk, raw, filt, pred);
        if (poseOk) drawAxes(frame, K, D, rvec, tvec, ARMOR_W_MM * 0.5f);

        imshow(WIN_NAME, frame);

        // ---- CSV ----
        if (csv && poseOk) {
            const double tms = chrono::duration<double, std::milli>(now - tStart).count();
            fprintf(csv, "%.1f,1,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    tms, raw.x, raw.y, raw.z, filt.x, filt.y, filt.z, pred.x, pred.y, pred.z);
        }

        ++frameCount;
        if (frameCount % 30 == 0) {
            printf("[time] fps %.1f | objects %zu\n", fpsNow, dets.size());
            if (poseOk) {
                const float d = std::sqrt(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z);
                printf("[pose] raw(%7.1f,%7.1f,%7.1f) filt(%7.1f,%7.1f,%7.1f) "
                       "pred(%7.1f,%7.1f,%7.1f) dist %.1f mm\n",
                       raw.x, raw.y, raw.z, filt.x, filt.y, filt.z,
                       pred.x, pred.y, pred.z, d);
            } else if (kalman.inited()) {
                printf("[pose] 本帧无检测，预测位置 (%7.1f,%7.1f,%7.1f)\n",
                       pred.x, pred.y, pred.z);
            }
        }

        const int key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    if (csv) fclose(csv);

    if (g_interrupted) printf("[MAIN] 收到 Ctrl+C，正在释放相机资源...\n");
    cam.close();
    destroyAllWindows();
    printf("[MAIN] 相机已关闭，共处理 %d 帧\n", frameCount);
    return 0;
}
