#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把主程序输出的 pose_log.csv 画成对比图，用于实验记录。

用法：
    python3 plot_pose.py                 # 读当前目录的 pose_log.csv
    python3 plot_pose.py 路径/xxx.csv    # 指定文件

输出：
    pose_compare.png     三联图：X / Y / Z 的 raw vs filt vs pred
    （终端同时打印 raw 与 filt 的抖动标准差，直接写进报告）

依赖：
    sudo apt install -y python3-matplotlib
    或 pip install matplotlib
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "pose_log.csv"
OUT = "pose_compare.png"

t, raw, filt, pred = [], [], [], []

with open(CSV, "r", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for row in rd:
        try:
            if int(row["detected"]) == 0:
                continue
            t.append(float(row["t_ms"]) / 1000.0)
            raw.append([float(row["raw_x"]),  float(row["raw_y"]),  float(row["raw_z"])])
            filt.append([float(row["filt_x"]), float(row["filt_y"]), float(row["filt_z"])])
            pred.append([float(row["pred_x"]), float(row["pred_y"]), float(row["pred_z"])])
        except (KeyError, ValueError):
            continue

if len(t) < 5:
    print("有效数据太少（%d 行），先跑一段再画" % len(t))
    sys.exit(1)

raw = np.array(raw)
filt = np.array(filt)
pred = np.array(pred)
t = np.array(t)

print("样本数：%d，时长 %.2f s" % (len(t), t[-1] - t[0]))
print("")
print("%-6s %12s %12s %10s" % ("轴", "raw 标准差", "filt 标准差", "降低"))
for i, ax in enumerate(["X", "Y", "Z"]):
    sr = raw[:, i].std()
    sf = filt[:, i].std()
    print("%-6s %12.2f %12.2f %9.0f%%" % (ax, sr, sf, (1 - sf / sr) * 100 if sr > 0 else 0))

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
names = ["X (mm)", "Y (mm)", "Z 距离 (mm)"]
for i in range(3):
    ax = axes[i]
    ax.plot(t, raw[:, i], color="#999999", lw=1.0, label="raw（PnP 直接解算）")
    ax.plot(t, filt[:, i], color="#1f77b4", lw=1.8, label="filt（卡尔曼滤波后）")
    ax.plot(t, pred[:, i], color="#d62728", lw=1.0, ls="--", alpha=0.8, label="pred（预测）")
    ax.set_ylabel(names[i])
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("时间 (s)")
fig.suptitle("装甲板三维位置：原始测量 vs 卡尔曼滤波 vs 预测")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("")
print("已保存：", os.path.abspath(OUT))
