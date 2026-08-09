"""
数据导入模块
==========
支持格式:
  - DEM 底图: GeoTIFF (.tif) [可选]
  - 测线数据: TXT (GPS + 反射强度) [必需]
  - SEPY 剖面数据
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from config.settings import TXT_DELIMITER, TXT_COLUMN_NAMES
from utils.logger import logger


@dataclass
class SurveyLine:
    """
    单条测线数据结构 (支持 SBP 道集)

    原始行: lon, lat, intensity, seq_number
    同一 (lon, lat) 位置的多行 = 该位置的道集采样 (不同深度层)
    """
    name: str = ""
    # 原始列数据
    longitude: np.ndarray = field(default_factory=lambda: np.array([]))
    latitude: np.ndarray = field(default_factory=lambda: np.array([]))
    reflection_intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    sequence_number: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_data: np.ndarray = field(default_factory=lambda: np.array([]))
    # 道集结构: 每个唯一 GPS → 该位置的多个 (seq, intensity)
    trace_positions: list = field(default_factory=list)  # [(lon, lat), ...]
    trace_samples: list = field(default_factory=list)    # [np.array([[seq, intensity],...]), ...]


@dataclass
class DEMData:
    """DEM 底图数据结构"""
    raster: np.ndarray = field(default_factory=lambda: np.array([]))
    transform: tuple = ()
    crs: str = ""
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    file_path: str = ""


@dataclass
class SepyProfile:
    """SEPY 剖面数据结构"""
    name: str = ""
    trace_data: np.ndarray = field(default_factory=lambda: np.array([]))
    num_traces: int = 0
    num_samples: int = 0
    sample_interval: float = 0.0
    trace_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    file_path: str = ""


def _haversine_simple(lon1, lat1, lon2, lat2):
    """两点间 Haversine 距离 (m)"""
    R = 6371000.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a+1e-15))


def _split_at_gps_jumps(lon, lat, intensity, seq, threshold_m=500):
    """
    在 GPS 大跳跃处拆分为独立测线。
    合并数据中不同测线之间 GPS 跳跃可达百公里，检测并切断。
    """
    if len(lon) < 2:
        return [(lon, lat, intensity, seq)]

    # 逐点计算相邻 GPS 距离
    jumps = np.zeros(len(lon))
    for i in range(1, len(lon)):
        jumps[i] = _haversine_simple(lon[i-1], lat[i-1], lon[i], lat[i])

    # 断点 = 跳跃 > threshold
    break_idx = np.where(jumps > threshold_m)[0]

    if len(break_idx) == 0:
        return [(lon, lat, intensity, seq)]

    # 切分
    sub_lines = []
    start = 0
    for bi in break_idx:
        if bi - start >= 5:  # 至少5个点才算有效测线
            sub_lines.append((lon[start:bi], lat[start:bi],
                              intensity[start:bi], seq[start:bi]))
        start = bi
    # 最后一段
    if len(lon) - start >= 5:
        sub_lines.append((lon[start:], lat[start:],
                          intensity[start:], seq[start:]))

    return sub_lines if sub_lines else [(lon, lat, intensity, seq)]


class DataLoader:
    """
    数据导入器
    =========
    统一管理 DEM、测线 TXT、SEPY 剖面数据的加载。
    可扩展：在此类中添加新的 _load_xxx 方法即可支持新格式。
    """

    def __init__(self):
        self.dem: Optional[DEMData] = None
        self.survey_lines: List[SurveyLine] = []
        self.sepy_profiles: List[SepyProfile] = []

    # ==================== DEM 底图 ====================

    def load_dem(self, file_path: str) -> Optional[DEMData]:
        """
        加载 GeoTIFF 格式 DEM 底图。

        Parameters
        ----------
        file_path : str
            .tif 文件路径

        Returns
        -------
        DEMData or None
        """
        if not os.path.isfile(file_path):
            logger.error(f"DEM 文件不存在: {file_path}")
            return None
        if not file_path.lower().endswith((".tif", ".tiff")):
            logger.error(f"DEM 文件格式不支持，请使用 GeoTIFF: {file_path}")
            return None

        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(file_path) as src:
                raster = src.read(1)
                transform = src.transform
                crs = str(src.crs)
                bounds = src.bounds

            # 转换为 WGS84 经纬度 (统一坐标系)
            try:
                if src.crs and src.crs.to_string() != "EPSG:4326":
                    wgs84_bounds = transform_bounds(src.crs, "EPSG:4326",
                        bounds.left, bounds.bottom, bounds.right, bounds.top)
                    bounds = (wgs84_bounds[0], wgs84_bounds[1],
                              wgs84_bounds[2], wgs84_bounds[3])
                    logger.info(f"DEM 坐标已转换: {src.crs} → EPSG:4326")
            except Exception:
                pass  # 转换失败则保留原始坐标

            self.dem = DEMData(
                raster=raster,
                transform=transform,
                crs=crs,
                bounds=bounds,
                file_path=file_path,
            )
            logger.info(f"DEM 加载成功: {file_path} 形状:{raster.shape} 范围:{bounds}")
            return self.dem

        except ImportError:
            # rasterio 未安装，用 PIL 作为只读图片的后备方案
            logger.warning("rasterio 未安装，尝试用 PIL 读取 (无地理参考)...")
            return self._load_dem_pil(file_path)

        except Exception as e:
            logger.warning(f"rasterio 打开失败，尝试 PIL 后备: {e}")
            return self._load_dem_pil(file_path)

    def _load_dem_pil(self, file_path: str) -> Optional[DEMData]:
        """PIL 后备方案: 读取 tif 图像数据 (无地理参考信息)"""
        try:
            from PIL import Image
            img = Image.open(file_path)
            raster = np.array(img)
            h, w = raster.shape[:2]

            self.dem = DEMData(
                raster=raster,
                transform=(),
                crs="unknown",
                bounds=(0, 0, w, h),
                file_path=file_path,
            )
            logger.info(f"DEM 加载成功 (PIL): {file_path} (形状: {raster.shape})")
            return self.dem

        except ImportError:
            logger.error("rasterio 和 Pillow 均未安装。请安装: pip install rasterio 或 pip install Pillow")
            return None
        except Exception as e:
            logger.exception(f"DEM 加载失败 (PIL): {e}")
            return None

    # ==================== 测线 TXT ====================

    def load_survey_txt(
        self,
        file_path: str,
        delimiter: str = None,
        has_header: bool = False,
        column_map: dict = None,
    ) -> List[SurveyLine]:
        """
        加载带 GPS 和反射强度的测线 TXT 数据。

        默认列顺序: 经度, 纬度, 深度, 反射强度
        可通过 column_map 自定义列映射:
            {0: 'lon', 1: 'lat', 2: 'depth', 3: 'intensity'}

        Parameters
        ----------
        file_path : str
            TXT 文件路径
        delimiter : str, optional
            分隔符，默认使用配置文件设置
        has_header : bool
            文件是否有标题行
        column_map : dict, optional
            列索引到字段的映射

        Returns
        -------
        list of SurveyLine
        """
        if delimiter is None:
            delimiter = TXT_DELIMITER

        if not os.path.isfile(file_path):
            logger.error(f"测线文件不存在: {file_path}")
            return []

        try:
            # 自动检测分隔符
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            if "\t" in first_line:
                delimiter = "\t"
            elif "," in first_line:
                delimiter = ","
            else:
                delimiter = None  # 空白分隔

            # 读取数据
            skiprows = 1 if has_header else 0
            raw = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)

            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 2:
                logger.error(f"测线数据列数不足 (至少需要 GPS 坐标): {file_path}")
                return []

            # 列映射 (默认匹配真实数据: lon, lat, intensity, seq)
            if column_map is None:
                col_lon = 0
                col_lat = 1
                col_intensity = 2 if raw.shape[1] > 2 else None
                col_seq = 3 if raw.shape[1] > 3 else None
            else:
                col_lon = column_map.get("lon", 0)
                col_lat = column_map.get("lat", 1)
                col_intensity = column_map.get("intensity", 2)
                col_seq = column_map.get("seq", 3)

            # 提取各列
            lon = raw[:, col_lon]
            lat = raw[:, col_lat]
            intensity = raw[:, col_intensity] if col_intensity is not None and col_intensity < raw.shape[1] else np.zeros_like(lon)
            seq = raw[:, col_seq] if col_seq is not None and col_seq < raw.shape[1] else np.arange(len(lon))

            # 自动拆分: GPS 跳跃 > 500m 处切为新测线
            sub_lines = _split_at_gps_jumps(lon, lat, intensity, seq, threshold_m=500)

            base_name = os.path.splitext(os.path.basename(file_path))[0]

            for sub_i, (s_lon, s_lat, s_int, s_seq) in enumerate(sub_lines):
                name = f"{base_name}" if len(sub_lines) == 1 else f"{base_name}_{sub_i+1}"
                survey_line = SurveyLine(
                    name=name,
                    longitude=s_lon,
                    latitude=s_lat,
                    reflection_intensity=s_int,
                    sequence_number=s_seq,
                    trace_positions=[], trace_samples=[],
                )
                self.survey_lines.append(survey_line)

            logger.info(
                f"测线加载成功: {file_path} → {len(sub_lines)} 条测线 "
                f"(总行数: {len(lon)}, "
                f"GPS: [{lon.min():.4f},{lon.max():.4f}] × [{lat.min():.4f},{lat.max():.4f}])"
            )
            return self.survey_lines

        except Exception as e:
            logger.exception(f"测线加载失败: {e}")
            return []

    # ==================== SEPY 剖面 ====================

    def load_sepy(self, file_path: str) -> Optional[SepyProfile]:
        """
        加载 SEPY 格式剖面数据。

        SEPY 是简化的地震/剖面数据格式:
          - 二进制或文本格式
          - 包含道集数据和位置信息

        此方法可供用户根据不同 SEPY 变体进行修改。

        Parameters
        ----------
        file_path : str
            SEPY 文件路径

        Returns
        -------
        SepyProfile or None
        """
        if not os.path.isfile(file_path):
            logger.error(f"SEPY 文件不存在: {file_path}")
            return None

        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ext = os.path.splitext(file_path)[1].lower()

            if ext in (".sgy", ".segy"):
                profile = self._load_segy(file_path)
            elif ext in (".txt", ".dat", ".csv"):
                profile = self._load_sepy_text(file_path)
            else:
                # 尝试二进制 SEG-Y
                profile = self._load_segy(file_path)

            if profile:
                profile.name = base_name
                profile.file_path = file_path
                self.sepy_profiles.append(profile)
                logger.info(
                    f"SEPY 剖面加载成功: {file_path} "
                    f"(道数: {profile.num_traces}, 采样点: {profile.num_samples})"
                )
            return profile

        except Exception as e:
            logger.exception(f"SEPY 加载失败: {e}")
            return None

    def _load_segy(self, file_path: str) -> SepyProfile:
        """加载 SEG-Y 格式。用户可替换为实际 SEG-Y 读取库。"""
        # 尝试使用 segyio 库
        try:
            import segyio
            with segyio.open(file_path, "r", strict=False) as f:
                num_traces = f.tracecount
                num_samples = f.samples.size if f.samples is not None else f.bin[segyio.BinField.Samples].size
                sample_interval = segyio.tools.dt(f, fallback=1000) / 1000.0  # ms -> s

                traces = np.zeros((num_traces, num_samples), dtype=np.float32)
                for i in range(num_traces):
                    traces[i] = f.trace[i]

                # 提取道位置
                trace_positions = np.zeros((num_traces, 2))
                for i in range(num_traces):
                    header = f.header[i]
                    trace_positions[i, 0] = header.get(segyio.TraceField.SourceX, i)
                    trace_positions[i, 1] = header.get(segyio.TraceField.SourceY, 0)

            return SepyProfile(
                trace_data=traces,
                num_traces=num_traces,
                num_samples=num_samples,
                sample_interval=sample_interval,
                trace_positions=trace_positions,
            )

        except ImportError:
            logger.warning("segyio 未安装，尝试二进制直接读取...")
            return self._load_sepy_binary(file_path)

    def _load_sepy_binary(self, file_path: str) -> SepyProfile:
        """二进制直读 SEG-Y（简化版，用户可根据实际格式修改）。"""
        with open(file_path, "rb") as f:
            raw = np.fromfile(f, dtype=np.float32)

        # 假设每条道 1024 个采样点
        num_samples = 1024
        num_traces = len(raw) // num_samples
        trace_data = raw[:num_traces * num_samples].reshape(num_traces, num_samples)
        trace_positions = np.arange(num_traces, dtype=np.float32).reshape(-1, 1)
        trace_positions = np.hstack([trace_positions, np.zeros((num_traces, 1))])

        return SepyProfile(
            trace_data=trace_data,
            num_traces=num_traces,
            num_samples=num_samples,
            sample_interval=1.0,
            trace_positions=trace_positions,
        )

    def _load_sepy_text(self, file_path: str) -> SepyProfile:
        """加载文本格式 SEPY 数据（矩阵形式，每行一道）。"""
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        num_traces = data.shape[0]
        num_samples = data.shape[1]
        trace_positions = np.arange(num_traces, dtype=np.float32).reshape(-1, 1)
        trace_positions = np.hstack([trace_positions, np.zeros((num_traces, 1))])

        return SepyProfile(
            trace_data=data,
            num_traces=num_traces,
            num_samples=num_samples,
            sample_interval=1.0,
            trace_positions=trace_positions,
        )

    # ==================== 辅助方法 ====================

    def get_all_survey_coordinates(self) -> Optional[np.ndarray]:
        """获取所有测线的合并坐标 (N, 2) -> [lon, lat]"""
        if not self.survey_lines:
            return None
        all_coords = []
        for sl in self.survey_lines:
            coords = np.column_stack([sl.longitude, sl.latitude])
            all_coords.append(coords)
        return np.vstack(all_coords)

    def get_all_intensity_data(self) -> Optional[np.ndarray]:
        """获取所有测线的反射强度与坐标合并数据"""
        if not self.survey_lines:
            return None
        all_data = []
        for sl in self.survey_lines:
            data = np.column_stack([sl.longitude, sl.latitude, sl.sequence_number, sl.reflection_intensity])
            all_data.append(data)
        return np.vstack(all_data)

    def clear(self):
        """清空所有已加载数据"""
        self.dem = None
        self.survey_lines.clear()
        self.sepy_profiles.clear()
        logger.info("已清空所有数据")


# ==================== 按需构建道集 ====================

def build_traces_in_window(
    lon: np.ndarray,
    lat: np.ndarray,
    intensity: np.ndarray,
    seq: np.ndarray,
    dist_min: float = None,
    dist_max: float = None,
) -> tuple:
    """
    在指定距离窗口内按需构建道集 (惰性，快速)。

    先按距离裁剪 → 再按 GPS 分组 → 每组按序号排序。
    只处理窗口内数据，适合大数据量的 SEGY 弹窗。

    Returns
    -------
    trace_positions : [(lon, lat), ...]
    trace_samples : [np.array([[seq, intensity], ...]), ...]
    """
    from ui.widgets.profile_view import _distance_along_track

    # 如果有距离约束，先裁剪
    if dist_min is not None and dist_max is not None:
        dist = _distance_along_track(lon, lat)
        mask = (dist >= dist_min) & (dist <= dist_max)
        lon = lon[mask]
        lat = lat[mask]
        intensity = intensity[mask]
        seq = seq[mask]

    n = len(lon)
    if n == 0:
        return [], []

    # 用 Python dict 快速分组 (O(n))
    groups = {}
    for i in range(n):
        key = (round(lon[i], 8), round(lat[i], 8))
        if key not in groups:
            groups[key] = []
        groups[key].append((seq[i], intensity[i]))

    # 转为 numpy 数组
    trace_positions = []
    trace_samples = []
    for (glon, glat), samples in groups.items():
        arr = np.array(samples)  # (n, 2) = (seq, intensity)
        # 按 seq 排序
        arr = arr[arr[:, 0].argsort()]
        trace_positions.append((glon, glat))
        trace_samples.append(arr)

    # 按经度排序
    order = np.argsort([tp[0] for tp in trace_positions])
    trace_positions = [trace_positions[i] for i in order]
    trace_samples = [trace_samples[i] for i in order]

    return trace_positions, trace_samples
