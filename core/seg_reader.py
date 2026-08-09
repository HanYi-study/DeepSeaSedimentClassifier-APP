"""
SEG-Y 浅剖数据读取器
===================
读取 SEG-Y 格式的浅地层剖面数据。

格式: 小端 IEEE float32, 固定 16400 采样点/道
"""

import os
import struct
import numpy as np
from dataclasses import dataclass


@dataclass
class SegData:
    """SEG-Y 数据"""
    name: str
    data: np.ndarray        # (n_traces, n_samples) float32 振幅矩阵
    n_traces: int
    n_samples: int
    sample_interval_us: int = 0
    coords: np.ndarray = None  # (n_traces, 2) [X, Y] 或 None


def read_seg_file(filepath: str) -> SegData:
    """
    读取单个 SEG 文件。

    Returns SegData with 2D amplitude matrix (n_traces × n_samples).
    """
    file_size = os.path.getsize(filepath)
    name = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, "rb") as f:
        # Text header (3200 bytes)
        f.seek(3200)

        # Binary header (400 bytes)
        bh = f.read(400)
        ns = struct.unpack('<h', bh[20:22])[0]  # samples per trace (little-endian)
        si = struct.unpack('<h', bh[16:18])[0]  # sample interval

        if ns <= 0 or ns > 100000:
            ns = 16400  # fallback

        # Trace header = 240 bytes
        trace_header_size = 240
        sample_size = 4  # float32

        total_header = 3600
        bytes_per_trace = trace_header_size + ns * sample_size
        remaining = file_size - total_header
        n_traces = remaining // bytes_per_trace

        if n_traces <= 0:
            n_traces = 1

        # Read all traces + header coords
        data = np.zeros((n_traces, ns), dtype=np.float32)
        coords = np.zeros((n_traces, 2), dtype=np.float64)
        has_coords = False

        for i in range(n_traces):
            # Read trace header
            f.seek(total_header + i * bytes_per_trace)
            th = f.read(trace_header_size)
            # SourceX at bytes 72-75, SourceY at 76-79 (big-endian int32)
            sx = struct.unpack('>i', th[72:76])[0]
            sy = struct.unpack('>i', th[76:80])[0]
            if sx != 0 or sy != 0:
                coords[i] = [sx, sy]
                has_coords = True
            else:
                coords[i] = [float(i), 0.0]  # fallback: relative pos

            # Read trace data
            f.seek(total_header + i * bytes_per_trace + trace_header_size)
            raw = f.read(ns * sample_size)
            data[i] = np.frombuffer(raw, dtype='<f4')

    return SegData(
        name=name, data=data, n_traces=n_traces,
        n_samples=ns, sample_interval_us=si,
        coords=coords if has_coords else None)


def read_seg_folder(folder_path: str) -> list:
    """
    读取整个文件夹的所有 SEG 文件。

    Returns list of SegData objects.
    """
    results = []
    if not os.path.isdir(folder_path):
        return results

    for fname in sorted(os.listdir(folder_path)):
        if fname.upper().endswith(".SEG") or fname.upper().endswith(".SGY"):
            fpath = os.path.join(folder_path, fname)
            try:
                seg = read_seg_file(fpath)
                results.append(seg)
                print(f"  Loaded: {seg.name} ({seg.n_traces} traces × {seg.n_samples} samples)")
            except Exception as e:
                print(f"  Failed: {fname}: {e}")

    return results
