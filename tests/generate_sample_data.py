#!/usr/bin/env python
"""
生成示例测试数据
===============
用于在没有真实数据时测试软件功能。

生成:
  1. sample_survey.txt  - 模拟测线数据 (GPS + 深度 + 反射强度)
  2. sample_profile.txt - 模拟 SEPY 剖面 (矩阵格式)
"""

import numpy as np
import os


def generate_survey_data(output_path="sample_survey.txt", n_points=500):
    """生成模拟的测线数据"""
    # 模拟南海北部陆坡区域
    lon_base = 110.5
    lat_base = 17.5

    # 5 条模拟测线，不同方向
    lines = []
    for i in range(5):
        n = n_points
        # 测线方向
        angle = i * 36 * np.pi / 180
        t = np.linspace(0, 0.05, n)
        lon = lon_base + t * np.cos(angle) + np.random.randn(n) * 0.001
        lat = lat_base + t * np.sin(angle) + np.random.randn(n) * 0.001

        # 深度: 100-500m, 缓变
        depth = 150 + 300 * np.sin(t * 30) + np.random.randn(n) * 5

        # 反射强度: 模拟5种底质的不同特征
        # 不同沉积物类型有不同的反射特征
        if i == 0:    # 钙质生物粉砂 - 低反射
            intensity = -35 + np.random.randn(n) * 3
        elif i == 1:  # 钙质生物黏土质粉砂 - 中低反射
            intensity = -30 + np.random.randn(n) * 3
        elif i == 2:  # 粉砂质砂 - 中等反射
            intensity = -25 + np.random.randn(n) * 4
        elif i == 3:  # 中砂 - 中高反射
            intensity = -20 + np.random.randn(n) * 3
        else:         # 砾砂 - 高反射
            intensity = -15 + np.random.randn(n) * 4

        # 写入
        with open(output_path, "a" if i > 0 else "w", encoding="utf-8") as f:
            if i == 0:
                f.write("# Longitude,Latitude,Depth(m),Reflection_Intensity(dB)\n")
            for j in range(n):
                f.write(f"{lon[j]:.6f},{lat[j]:.6f},{depth[j]:.2f},{intensity[j]:.2f}\n")

    print(f"[OK] 测线数据已生成: {output_path} ({n_points * 5} 点, 5 条测线)")

    # 同时生成每条独立测线的多文件版本
    for i in range(5):
        line_path = f"sample_survey_line_{i+1}.txt"
        # 重新生成一条独立测线
        n = n_points
        angle = i * 36 * np.pi / 180
        t = np.linspace(0, 0.05, n)
        lon = lon_base + t * np.cos(angle) + np.random.randn(n) * 0.001
        lat = lat_base + t * np.sin(angle) + np.random.randn(n) * 0.001
        depth = 150 + 300 * np.sin(t * 30) + np.random.randn(n) * 5
        if i == 0:
            intensity = -35 + np.random.randn(n) * 3
        elif i == 1:
            intensity = -30 + np.random.randn(n) * 3
        elif i == 2:
            intensity = -25 + np.random.randn(n) * 4
        elif i == 3:
            intensity = -20 + np.random.randn(n) * 3
        else:
            intensity = -15 + np.random.randn(n) * 4

        with open(line_path, "w", encoding="utf-8") as f:
            f.write("# Longitude,Latitude,Depth(m),Reflection_Intensity(dB)\n")
            for j in range(n):
                f.write(f"{lon[j]:.6f},{lat[j]:.6f},{depth[j]:.2f},{intensity[j]:.2f}\n")
        print(f"  -> {line_path}")


def generate_sepy_profile(output_path="sample_profile.txt", n_traces=256, n_samples=1024):
    """生成模拟 SEPY 剖面数据 (文本矩阵格式)"""
    # 模拟海底地层反射
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)

    # 添加背景噪声
    traces += np.random.randn(n_traces, n_samples) * 0.05

    # 添加海底反射层 (在样本点 100 附近)
    seabed_idx = 100
    for i in range(n_traces):
        # 海底深度稍有变化
        shift = int(np.sin(i / n_traces * 4 * np.pi) * 10)
        idx = seabed_idx + shift
        if 0 <= idx < n_samples:
            traces[i, idx] = 1.0 + np.random.randn() * 0.1

    # 添加次表层反射层
    subsurface_idx = 200
    for i in range(n_traces):
        shift = int(np.sin(i / n_traces * 3 * np.pi) * 15)
        idx = subsurface_idx + shift
        if 0 <= idx < n_samples:
            traces[i, idx] = 0.6 + np.random.randn() * 0.1

    # 添加深层反射
    deep_idx = 400
    for i in range(n_traces):
        shift = int(np.cos(i / n_traces * 2 * np.pi) * 20)
        idx = deep_idx + shift
        if 0 <= idx < n_samples:
            traces[i, idx] = 0.3 + np.random.randn() * 0.1

    # 保存为文本矩阵 (每行一道)
    np.savetxt(output_path, traces, fmt="%.4f")
    print(f"[OK] SEPY 剖面数据已生成: {output_path} ({n_traces} 道 x {n_samples} 采样点)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("生成示例测试数据...")
    print("=" * 50)
    generate_survey_data()
    print()
    generate_sepy_profile()
    print("=" * 50)
    print("生成完成! 可在软件中加载以下文件进行测试:")
    print("  1. sample_survey.txt          - 测线数据")
    print("  2. sample_survey_line_1~5.txt - 单条测线")
    print("  3. sample_profile.txt         - SEPY 剖面")
