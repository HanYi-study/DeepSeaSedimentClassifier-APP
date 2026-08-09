"""
SEG 浅剖数据预处理与去噪
======================
对 SEG-Y 道集数据进行预处理, 提升分类特征质量。

方法:
  1. 中值滤波: 去除尖峰/野值噪声
  2. 高斯平滑: 去除高频随机噪声
  3. 带通滤波: 保留有效频带, 去除低频漂移和高频噪声
  4. AGC (自动增益控制): 均衡深浅层振幅
  5. 去均值: 去除直流分量
  6. 道间均衡: 均衡不同道的能量差异
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import butter, filtfilt


def remove_mean(data):
    """去直流分量: 每道减去均值"""
    return data - np.mean(data, axis=1, keepdims=True)


def median_denoise(data, kernel_size=5):
    """中值滤波去尖峰噪声"""
    return median_filter(data, size=(1, kernel_size))


def gaussian_denoise(data, sigma=2.0):
    """高斯平滑去高频噪声"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        result[i] = gaussian_filter1d(data[i].astype(float), sigma=sigma)
    return result


def bandpass_filter(data, lowcut=50, highcut=5000, fs=16400):
    """带通滤波: 保留 [lowcut, highcut] 频率范围"""
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.99
    b, a = butter(4, [low, high], btype='band')
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        result[i] = filtfilt(b, a, data[i])
    return result


def agc(data, window=200):
    """自动增益控制: 滑动窗口均衡振幅"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        trace = data[i].copy()
        # 滑动窗口 RMS
        rms = np.zeros(len(trace))
        half = window // 2
        for j in range(len(trace)):
            lo = max(0, j - half)
            hi = min(len(trace), j + half)
            rms[j] = np.sqrt(np.mean(trace[lo:hi] ** 2)) + 1e-10
        result[i] = trace / rms
    return result


def trace_balance(data):
    """道间均衡: 每道除以其 RMS, 消除道间能量差异"""
    rms = np.sqrt(np.mean(data ** 2, axis=1, keepdims=True)) + 1e-10
    return data / rms


def auto_detect_methods(data):
    """
    根据数据特征自动选择去噪方法。

    SEG 数据 (n_traces × n_samples, n_samples > 10):
      → 去均值 + 中值滤波 + AGC
    TXT 数据 (1D array):
      → 中值滤波 + 高斯平滑
    """
    if data.ndim == 2 and data.shape[1] > 10:
        # SEG/SBP 道集数据
        std_ratio = np.std(data) / (np.mean(np.abs(data)) + 1e-10)
        steps = ['demean']
        if std_ratio > 3:
            steps.append('median')  # 有尖峰噪声
        steps.append('agc')
        # 如果高频噪声明显
        diff_std = np.std(np.diff(data, axis=1))
        if diff_std > np.std(data) * 2:
            steps.append('gaussian')
        return steps
    else:
        # TXT 点数据
        steps = ['median']
        # 小波动 → 加高斯平滑
        if len(data) > 10:
            local_std = np.std(np.diff(data))
            if local_std > np.std(data) * 0.5:
                steps.append('gaussian')
        return steps


def preprocess_pipeline(data, steps=None):
    """
    SEG 预处理流水线。steps=None 时自动检测。

    Parameters
    ----------
    data : (n_traces, n_samples) ndarray
    steps : list of str, None=自动选择

    Returns
    -------
    (processed_2d, steps_list)
    """
    if steps is None:
        steps = auto_detect_methods(data)

    result = data.astype(np.float64).copy()
    for step in steps:
        if step == 'demean':
            result = remove_mean(result)
        elif step == 'median':
            result = median_denoise(result, kernel_size=5)
        elif step == 'gaussian':
            result = gaussian_denoise(result, sigma=2.0)
        elif step == 'bandpass':
            result = bandpass_filter(result)
        elif step == 'agc':
            result = agc(result, window=200)
        elif step == 'balance':
            result = trace_balance(result)
    return result.astype(np.float32), list(steps)
