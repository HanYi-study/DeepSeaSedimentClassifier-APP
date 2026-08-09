"""
SEG 浅剖数据 → 分类特征提取
=========================
从 SEG-Y 道集数据中提取分类特征, 用于 MSC-Transformer 训练。

论文方法 (JMSE 2023):
  SBP 道集 → 振幅统计 + 频谱 + 纹理 → 输入 Transformer

每道提取的特征:
  1. 振幅统计: mean, std, max, min, skewness, kurtosis
  2. 能量特征: 总能量, 峰值能量位置(海底深度代理)
  3. 频率特征: 过零率, 自相关峰值
  4. 分层特征: 前20%采样 vs 后80%采样的能量比
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def extract_seg_features(seg_data_list: list, max_traces_per_file: int = 5000):
    """
    从 SEG 数据列表提取分类特征。

    Parameters
    ----------
    seg_data_list : list of SegData
    max_traces_per_file : 每文件最大道数 (控制总样本量)

    Returns
    -------
    features : (N, F) 特征矩阵
    coords : (N, 2) 占位坐标 (无GPS时用文件索引)
    """
    all_features = []
    all_coords = []

    for file_idx, seg in enumerate(seg_data_list):
        data = seg.data
        n_traces, n_samples = data.shape

        # 坐标: 优先用道头 GPS, 否则相对位置
        if seg.coords is not None:
            file_coords = seg.coords
        else:
            file_coords = np.column_stack([
                np.full(n_traces, file_idx, dtype=float),
                np.arange(n_traces, dtype=float),
            ])

        # 抽样限制
        step = max(1, n_traces // max_traces_per_file)
        indices = range(0, n_traces, step)
        n_used = len(indices)

        for t in indices:
            trace = data[t].astype(np.float64)
            all_coords.append(file_coords[t])

            # ---- 振幅统计 ----
            mean_amp = np.mean(trace)
            std_amp = np.std(trace)
            max_amp = np.max(np.abs(trace))
            rms = np.sqrt(np.mean(trace ** 2))

            # 偏度和峰度 (需要 scipy)
            try:
                sk = stats.skew(trace)
                kt = stats.kurtosis(trace)
            except Exception:
                sk, kt = 0.0, 0.0

            # ---- 能量特征 ----
            energy = np.sum(trace ** 2)
            # 峰值位置 (归一化到 [0,1])
            peak_idx = np.argmax(np.abs(trace))
            peak_pos = peak_idx / n_samples

            # ---- 过零率 (频率代理) ----
            zero_crossings = np.sum(np.diff(np.signbit(trace))) / n_samples

            # ---- 分层能量比 (浅层 vs 深层) ----
            split = n_samples // 5
            shallow_energy = np.sum(trace[:split] ** 2) / split
            deep_energy = np.sum(trace[split:] ** 2) / (n_samples - split) if split < n_samples else shallow_energy
            energy_ratio = shallow_energy / (deep_energy + 1e-10)

            # ---- 平滑后梯度 (反射界面强度) ----
            smoothed = gaussian_filter1d(np.abs(trace), sigma=5)
            gradient = np.mean(np.abs(np.diff(smoothed)))

            features = [
                mean_amp, std_amp, max_amp, rms,
                sk, kt,
                energy, peak_pos, zero_crossings,
                energy_ratio, gradient,
                float(file_idx),  # 文件索引 (用于区分不同测线)
                float(t) / n_traces,  # 道在文件中的相对位置
            ]

            all_features.append(features)

        print(f"  {seg.name}: {n_used} traces → 13 features each, coords={'GPS' if seg.coords is not None else 'relative'}")

    return np.array(all_features, dtype=np.float32), np.array(all_coords, dtype=np.float32)
