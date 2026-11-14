// ============================================================================
//  calibrate.cpp —— 相机标定工具（第五次作业 任务 1）
//
//  用法：
//      ./Calibrate                    # 默认棋盘格 9x6 内角点、方格 25mm
//      ./Calibrate --cols 9 --rows 6 --square 25
//
//  操作：
//      1. 把棋盘格标定板放在相机前，变换不同位置与角度（建议 15~20 张）
//      2. 画面中出现彩色角点 + 绿色框 = 识别成功，按 空格 保存一帧
//      3. 至少采集 10 张后再按 ESC 结束，程序自动标定并写入 camera.yaml
//      4. 尽量覆盖：左/中/右、上/中/下、远/近、倾斜/正对
//
//  输出：
//      camera.yaml   内参矩阵 + 畸变系数 + 分辨率 + 重投影误差
//      calib_imgs/   采集的原始图像（留作实验记录）
//
//  注意：标定用的分辨率必须和检测程序运行时一致（本程序不做 ROI 裁剪）
// ============================================================================

#include <MvCameraControl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
//  棋盘格参数（实测：9x6 内角点，格子边长 26mm）
//  BOARD_COLS / BOARD_ROWS 是「内角点」个数，不是方格数：
//    一块 10x7 方格的板子 -> 9x6 个内角点
// ---------------------------------------------------------------------------
static int   BOARD_COLS = 9;
static int   BOARD_ROWS = 6;
static float SQUARE_MM  = 26.0f;

// 曝光时间（µs），可用 --exposure 覆盖，方便调试亮度
static int   g_exposureUs = 8000;

static const char* WIN_NAME = "Calibration";
static const char* OUT_YAML = "camera.yaml";
static const char* IMG_DIR  = "calib_imgs";

static string camString(void* h, const char* key)
{
    MVCC_STRINGVALUE sv;
    memset(&sv, 0, sizeof(sv));
    sv.nMaxLength = (int64_t)sizeof(sv.chCurValue);
    if (MV_CC_GetStringValue(h, key, &sv) != MV_OK) return string("N/A");
    return string(sv.chCurValue);
}

class HikCamera
{
public:
    ~HikCamera() { close(); }

    bool open()
    {
        MV_CC_DEVICE_INFO_LIST list;
        memset(&list, 0, sizeof(list));

        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
        if (ret != MV_OK || list.nDeviceNum == 0) {
            printf("[MVS] 没枚举到相机（检查连接、sudo、udev 规则）\n");
            return false;
        }
        ret = MV_CC_CreateHandle(&handle_, list.pDeviceInfo[0]);
        if (ret != MV_OK) { printf("[MVS] CreateHandle 失败 0x%X\n", (unsigned)ret); handle_ = nullptr; return false; }

        ret = MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0);
        if (ret != MV_OK) { printf("[MVS] OpenDevice 失败 0x%X（可能需 sudo）\n", (unsigned)ret); close(); return false; }

        printf("[INFO] 相机型号 : %s\n", camString(handle_, "DeviceModelName").c_str());
        printf("[INFO] 序列号   : %s\n", camString(handle_, "DeviceSerialNumber").c_str());
        printf("[INFO] 固件版本 : %s\n", camString(handle_, "DeviceVersion").c_str());

        // ---- 采集参数：与主程序保持一致，并且检查返回值 ----
        MV_CC_SetEnumValue(handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);
        MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        MV_CC_SetEnumValue(handle_, "GainAuto",     MV_GAIN_MODE_OFF);

        ret = MV_CC_SetFloatValue(handle_, "ExposureTime", (float)g_exposureUs);
        printf("[INFO] 设置曝光   : ret = 0x%X %s\n", (unsigned)ret,
               ret == MV_OK ? "OK" : "失败");
        ret = MV_CC_SetFloatValue(handle_, "Gain", 0.0f);
        printf("[INFO] 设置增益   : ret = 0x%X %s\n", (unsigned)ret,
               ret == MV_OK ? "OK" : "失败");

        ret = MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
        printf("[INFO] 设置BGR8    : ret = 0x%X %s\n", (unsigned)ret,
               ret == MV_OK ? "OK" : "失败");

        // ---- 回读：看相机实际生效的值（比设置返回值更能说明问题）----
        MVCC_FLOATVALUE fv;
        memset(&fv, 0, sizeof(fv));
        if (MV_CC_GetFloatValue(handle_, "ExposureTime", &fv) == MV_OK) {
            printf("[INFO] 实际曝光   : %.1f us  (范围 %.1f ~ %.1f)\n",
                   fv.fCurValue, fv.fMin, fv.fMax);
        } else {
            printf("[INFO] 实际曝光   : 回读失败\n");
        }

        MVCC_ENUMVALUE ev;
        memset(&ev, 0, sizeof(ev));
        if (MV_CC_GetEnumValue(handle_, "PixelFormat", &ev) == MV_OK) {
            printf("[INFO] 实际格式   : 0x%08X  (BGR8=0x02180015 Mono8=0x01080001 BayerRG8=0x01080009)\n",
                   ev.nCurValue);
        } else {
            printf("[INFO] 实际格式   : 回读失败\n");
        }

        ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) { printf("[MVS] StartGrabbing 失败 0x%X\n", (unsigned)ret); close(); return false; }
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

        lastPixelType = (unsigned int)fo.stFrameInfo.enPixelType;

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

    unsigned int lastPixelType = 0;

private:
    void*                 handle_   = nullptr;
    bool                  grabbing_ = false;
    vector<unsigned char> buf_;
};

// Ctrl+C 也能正常收尾（不然按不了 ESC 时只能强杀，白采）
static volatile sig_atomic_t g_stop = 0;
static void onSigInt(int) { g_stop = 1; }

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    signal(SIGINT, onSigInt);

    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--cols" && i + 1 < argc)        BOARD_COLS = atoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc)   BOARD_ROWS = atoi(argv[++i]);
        else if (a == "--square" && i + 1 < argc) SQUARE_MM  = (float)atof(argv[++i]);
        else if (a == "--exposure" && i + 1 < argc) g_exposureUs = atoi(argv[++i]);
    }

    printf("===== 相机标定 =====\n");
    printf("[CFG] 棋盘格内角点: %d x %d | 方格边长: %.1f mm | 曝光 %d us\n",
           BOARD_COLS, BOARD_ROWS, SQUARE_MM, g_exposureUs);

    HikCamera cam;
    if (!cam.open()) return -1;

    namedWindow(WIN_NAME, WINDOW_AUTOSIZE);

    // 三维物点：棋盘格平面 z=0
    vector<Point3f> objTemplate;
    for (int r = 0; r < BOARD_ROWS; ++r)
        for (int c = 0; c < BOARD_COLS; ++c)
            objTemplate.push_back(Point3f(c * SQUARE_MM, r * SQUARE_MM, 0.0f));

    vector<vector<Point3f>> objPoints;
    vector<vector<Point2f>> imgPoints;
    Size boardSize(BOARD_COLS, BOARD_ROWS);
    Size imageSize;

    // 保存图像的目录
    system(("mkdir -p " + string(IMG_DIR)).c_str());

    int saved = 0;
    bool running = true;
    int  diagCount = 0;

    while (running && !g_stop) {
        Mat frame;
        if (!cam.grab(frame)) { if ((waitKey(1) & 0xFF) == 27) break; continue; }
        imageSize = frame.size();

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ---- 亮度诊断：画面全黑时直接提示，别再猜 ----
        const double meanV = mean(gray)[0];
        if (++diagCount % 60 == 0) {
            printf("[DIAG] 平均灰度 %.1f | 像素格式 0x%08X | 分辨率 %dx%d\n",
                   meanV, cam.lastPixelType, frame.cols, frame.rows);
        }
        if (meanV < 8.0) {
            putText(frame, "TOO DARK: lens cap / aperture / exposure / light",
                    Point(15, 75), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);
        }

        // ---- 角点检测 ----
        // 在缩小图上做检测（速度快约 4 倍），亚像素精化仍在原图上做，精度不损失。
        // 同时去掉 CALIB_CB_NORMALIZE_IMAGE：暗图上直方图均衡非常慢。
        const double kScale = (gray.cols > 900) ? 0.5 : 1.0;
        Mat det;
        if (kScale != 1.0) resize(gray, det, Size(), kScale, kScale);
        else               det = gray;

        vector<Point2f> corners;
        bool found = findChessboardCorners(
            det, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK);

        if (found) {
            if (kScale != 1.0) {
                for (size_t i = 0; i < corners.size(); ++i) {
                    corners[i].x = (float)(corners[i].x / kScale);
                    corners[i].y = (float)(corners[i].y / kScale);
                }
            }
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));
            drawChessboardCorners(frame, boardSize, corners, found);
        }

        // HUD
        char hud[256];
        snprintf(hud, sizeof(hud), "views: %d | %s | SPACE=save  ESC=finish",
                 saved, found ? "DETECTED" : "searching...");
        putText(frame, hud, Point(15, 35), FONT_HERSHEY_SIMPLEX, 0.8,
                found ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2, LINE_AA);

        imshow(WIN_NAME, frame);
        const int key = waitKey(1) & 0xFF;

        if (key == ' ' && found) {
            objPoints.push_back(objTemplate);
            imgPoints.push_back(corners);
            ++saved;

            char fn[64];
            snprintf(fn, sizeof(fn), "%s/view_%02d.png", IMG_DIR, saved);
            imwrite(fn, frame);
            printf("[CAP] 已保存第 %d 张: %s\n", saved, fn);
        } else if (key == 27 || key == 'q' || key == 'Q') {
            // 焦点不在图像窗口时 ESC 收不到，q 是备用；Ctrl+C 由信号处理兜底
            printf("[CAP] 结束采集，共 %d 张\n", saved);
            running = false;
        }
    }

    cam.close();
    destroyAllWindows();

    if (saved < 8) {
        printf("[ERR] 只采集了 %d 张，至少需要 8 张（建议 15 张以上）才能标定\n", saved);
        return -1;
    }

    printf("[CAL] 开始标定，%d 张图像，分辨率 %dx%d ...\n", saved, imageSize.width, imageSize.height);

    Mat K, D;
    vector<Mat> rvecs, tvecs;
    double rms = calibrateCamera(objPoints, imgPoints, imageSize, K, D, rvecs, tvecs,
                                 CALIB_FIX_K3);   // 只估 k1,k2,p1,p2

    printf("\n===== 标定结果 =====\n");
    printf("重投影误差 RMS : %.4f px\n", rms);
    printf("内参矩阵 K:\n");
    printf("  [ %10.3f  %10.3f  %10.3f ]\n", K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2));
    printf("  [ %10.3f  %10.3f  %10.3f ]\n", K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2));
    printf("  [ %10.3f  %10.3f  %10.3f ]\n", K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2));
    printf("畸变系数 D     : %.6f  %.6f  %.6f  %.6f  %.6f\n",
           D.at<double>(0), D.at<double>(1), D.at<double>(2), D.at<double>(3), D.at<double>(4));
    printf("fx = %.3f, fy = %.3f, cx = %.3f, cy = %.3f\n",
           K.at<double>(0,0), K.at<double>(1,1), K.at<double>(0,2), K.at<double>(1,2));

    FileStorage fs(OUT_YAML, FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_cols" << BOARD_COLS;
    fs << "board_rows" << BOARD_ROWS;
    fs << "square_mm" << SQUARE_MM;
    fs << "rms" << rms;
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << D;
    fs.release();

    printf("\n[OK] 已写入 %s，检测程序会自动加载\n", OUT_YAML);
    return 0;
}
